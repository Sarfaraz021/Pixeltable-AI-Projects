import os
import pixeltable as pxt
from pixeltable.iterators.document import DocumentSplitter
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions import openai
import numpy as np

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = "sk-proj-NSq5V9uOwBJFpbz7NojMT3BlbkFJvZ3jq4W7pG7HZJGSLA9c"

# Initialize tables and views


def initialize_tables():
    pxt.drop_table('rag_demo2.chunks', ignore_errors=True)
    pxt.drop_table('rag_demo2.documents', ignore_errors=True)
    pxt.drop_table('rag_demo2.queries', ignore_errors=True, force=True)
    pxt.drop_table('rag_demo2.docs', ignore_errors=True)
    pxt.create_dir('rag_demo2', ignore_errors=True)

    documents_t = pxt.create_table('rag_demo2.documents', {
                                   'document': pxt.DocumentType()})
    file_path = r"D:\Pixeltable-AI-Projects\Data\cot.txt"
    documents_t.insert([{'document': file_path}])

    chunks_t = pxt.create_view(
        'rag_demo2.chunks',
        documents_t,
        iterator=DocumentSplitter.create(
            document=documents_t.document, separators='token_limit', limit=300)
    )

    @pxt.expr_udf
    def e5_embed(text: str) -> np.ndarray:
        return sentence_transformer(text, model_id='intfloat/e5-large-v2')

    chunks_t.add_embedding_index('text', text_embed=e5_embed)

    queries_t = pxt.create_table(
        'rag_demo2.queries', {'question': pxt.StringType()})

    return chunks_t, queries_t

# Set up query functions


def setup_query_functions(chunks_t, queries_t):
    @chunks_t.query
    def top_k(query_text: str):
        sim = chunks_t.text.similarity(query_text)
        return (
            chunks_t.order_by(sim, asc=False)
            .select(chunks_t.text, sim=sim)
            .limit(5)
        )

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

    messages = [
        {'role': 'system', 'content': 'Please read the following passages and answer the question based on their contents.'},
        {'role': 'user', 'content': queries_t.prompt}
    ]

    queries_t['response'] = openai.chat_completions(
        model='gpt-4o', messages=messages)
    queries_t['answer'] = queries_t.response.choices[0].message.content

# Main loop for asking questions


def main():
    chunks_t, queries_t = initialize_tables()
    setup_query_functions(chunks_t, queries_t)

    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        queries_t.insert([{'question': question}])
        response = queries_t.select(queries_t.answer).show()
        print("Answer:", response[0]['answer'])
        print()

    print("Thank you for using the system. Goodbye!")


if __name__ == "__main__":
    main()
