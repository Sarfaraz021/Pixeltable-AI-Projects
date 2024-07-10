import urllib
from pixeltable.iterators.document import DocumentSplitter
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions import openai
import pixeltable as pxt
import numpy as np
import urllib.request
import logging
import os
# from dotenv import load_dotenv
# load_dotenv("var.env")
# os.getenv("OPENAI_API_KEY")

os.environ['OPENAI_API_KEY'] = "sk-proj-NSq5V9uOwBJFpbz7NojMT3BlbkFJvZ3jq4W7pG7HZJGSLA9c"

# Make sure we start with a clean slate
pxt.drop_table('rag_demo2.chunks', ignore_errors=True)
pxt.drop_table('rag_demo2.documents', ignore_errors=True)
pxt.drop_table('rag_demo2.queries', ignore_errors=True, force=True)
pxt.drop_table('rag_demo2.docs', ignore_errors=True)

pxt.create_dir('rag_demo2', ignore_errors=True)


docs = pxt.create_table('rag_demo2.docs', {
    'source_doc': pxt.DocumentType()
})

# Local file path of the single PDF file
file_path = r"D:\Pixeltable-AI-Projects\Data\COT.pdf"

# Create a table to store the document
documents_t = pxt.create_table(
    'rag_demo2.documents', {'document': pxt.DocumentType()}
)

# Insert the single PDF document into the table
try:
    documents_t.insert([{'document': file_path}])
    print("Document inserted successfully.")
except pxt.exceptions.Error as e:
    print(f"Error inserting document: {e}")

# Create a view to split documents into chunks
chunks_t = pxt.create_view(
    'rag_demo2.chunks',
    documents_t,
    iterator=DocumentSplitter.create(
        document=documents_t.document,
        separators='token_limit', limit=300
    )
)


@pxt.expr_udf
def e5_embed(text: str) -> np.ndarray:
    return sentence_transformer(text, model_id='intfloat/e5-large-v2')


chunks_t.add_embedding_index('text', text_embed=e5_embed)

# Create a table to store the queries with a column for the question
pxt.drop_table('rag_demo2.queries', ignore_errors=True)
queries_t = pxt.create_table(
    'rag_demo2.queries', {'question': pxt.StringType()})


@chunks_t.query
def top_k(query_text: str):
    sim = chunks_t.text.similarity(query_text)
    return (
        chunks_t.order_by(sim, asc=False)
        .select(chunks_t.text, sim=sim)
        .limit(5)
    )


# Define the custom query
question = "what is chain of thought?"

# Insert the custom query into the queries_t table
queries_t.insert([{'question': question}])

queries_t['question_context'] = chunks_t.top_k(queries_t.question)


@pxt.udf
def create_prompt(top_k_list: list[dict], question: str) -> str:
    concat_top_k = '\n\n'.join(elt['text'] for elt in reversed(top_k_list))
    return f'''
    PASSAGES:

    {concat_top_k}

    QUESTION:

    {question}'''


queries_t['prompt'] = create_prompt(
    queries_t.question_context, queries_t.question)

# Assemble the prompt and instructions into OpenAI's message format
messages = [
    {'role': 'system', 'content': 'Please read the following passages and answer the question based on their contents.'},
    {'role': 'user', 'content': queries_t.prompt}
]

# Add a computed column that calls OpenAI
queries_t['response'] = openai.chat_completions(
    model='gpt-4o', messages=messages)

queries_t['answer'] = queries_t.response.choices[0].message.content

response = queries_t.select(queries_t.answer).show()
print(response)
