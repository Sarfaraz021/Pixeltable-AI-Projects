import os
import pixeltable as pxt
from pixeltable.iterators.document import DocumentSplitter
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions import openai
import numpy as np

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = "sk-proj-NSq5V9uOwBJFpbz7NojMT3BlbkFJvZ3jq4W7pG7HZJGSLA9c"


def initialize_tables():
    pxt.drop_table('rag_demo2.chunks', ignore_errors=True)
    pxt.drop_table('rag_demo2.documents', ignore_errors=True)
    pxt.drop_table('rag_demo2.queries', ignore_errors=True, force=True)
    pxt.drop_table('rag_demo2.docs', ignore_errors=True)
    pxt.create_dir('rag_demo2', ignore_errors=True)

    documents_t = pxt.create_table('rag_demo2.documents', {
        'document': pxt.DocumentType()
    })
    file_path = r"D:\Pixeltable-AI-Projects\Data\cot.txt"
    documents_t.insert([{'document': file_path}])

    chunks_t = pxt.create_view(
        'rag_demo2.chunks',
        documents_t,
        iterator=DocumentSplitter.create(
            document=documents_t.document, separators='token_limit', limit=300
        )
    )

    @pxt.expr_udf
    def e5_embed(text: str) -> np.ndarray:
        return sentence_transformer(text, model_id='intfloat/e5-large-v2')

    chunks_t.add_embedding_index('text', text_embed=e5_embed)

    queries_t = pxt.create_table(
        'rag_demo2.queries', {'question': pxt.StringType()}
    )

    return chunks_t, queries_t


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
        concat_top_k = '\n\n'.join(
            f"Passage {i+1}:\n{elt['text']}" for i, elt in enumerate(top_k_list))
        return f'''Use the following passages to answer the question. If the answer cannot be found in the passages, say "I don't have enough information to answer this question."

{concat_top_k}

Question: {question}

Answer:'''

    queries_t['prompt'] = create_prompt(
        queries_t.question_context, queries_t.question
    )

    messages = [
        {'role': 'system', 'content': 'You are a helpful AI assistant. Answer questions based solely on the given context. If the answer is not in the context, say you don\'t have enough information.'},
        {'role': 'user', 'content': queries_t.prompt}
    ]

    queries_t['response'] = openai.chat_completions(
        model='gpt-4o',  # Changed from 'gpt-4' to 'gpt-3.5-turbo'
        messages=messages,
        temperature=0.7  # Added temperature parameter
    )
    queries_t['answer'] = queries_t.response.choices[0].message.content


def main():
    chunks_t, queries_t = initialize_tables()
    setup_query_functions(chunks_t, queries_t)

    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        queries_t.insert([{'question': question}])
        response = queries_t.select(queries_t.prompt, queries_t.answer).show()

        print("\nPrompt:")
        print(response[0]['prompt'])
        print("\nAnswer:", response[0]['answer'])

        # Debug information
        print("\nDebug Info:")
        context = queries_t.select(queries_t.question_context).show()
        print("Top 5 relevant passages:")
        for i, passage in enumerate(context[0]['question_context']):
            print(f"Passage {i+1} (similarity: {passage['sim']:.4f}):")
            print(passage['text'])
            print()

    print("Thank you for using the system. Goodbye!")


if __name__ == "__main__":
    main()
