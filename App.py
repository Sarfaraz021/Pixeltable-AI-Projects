import streamlit as st
import os
import pixeltable as pxt
from pixeltable.iterators.document import DocumentSplitter
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions import openai
import numpy as np
from datetime import datetime
import tempfile
from dotenv import load_dotenv
# Set up OpenAI API key
load_dotenv("var.env")
os.getenv("OPENAI_API_KEY")

# Initialize tables and views


def initialize_tables(file_path):
    pxt.create_dir('rag_demo2', ignore_errors=True)

    # Try to drop existing tables
    tables_to_drop = ['rag_demo2.chunks', 'rag_demo2.documents',
                      'rag_demo2.queries', 'rag_demo2.docs']
    for table_name in tables_to_drop:
        try:
            pxt.drop_table(table_name, force=True)
        except Exception:
            pass  # If the table doesn't exist, just continue

    try:
        documents_t = pxt.create_table('rag_demo2.documents', {
                                       'document': pxt.DocumentType()})
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
            'rag_demo2.queries', {
                'id': pxt.IntType(),
                'question': pxt.StringType(),
                'timestamp': pxt.TimestampType()
            })

        return chunks_t, queries_t
    except Exception as e:
        raise Exception(f"Error initializing tables: {str(e)}")

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
        {'role': 'system', 'content': 'You are a helpful AI assistant. Answer questions based solely on the given context.'},
        {'role': 'user', 'content': queries_t.prompt}
    ]

    queries_t['response'] = openai.chat_completions(
        model='gpt-4o',
        messages=messages,
        temperature=0.7)
    queries_t['answer'] = queries_t.response.choices[0].message.content


def main():
    st.set_page_config(page_title="KnowledgeGPT", page_icon="📚")
    st.title("📚 AskPDF")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Initialize tables with the uploaded file
            chunks_t, queries_t = initialize_tables(tmp_file_path)
            setup_query_functions(chunks_t, queries_t)

            st.success("File uploaded and processed successfully!")

            # Chat interface
            st.subheader("Chat with your Docs")
            question = st.text_input("Enter your question:")

            if st.button("Ask"):
                if question:
                    with st.spinner("Generating answer..."):
                        # Insert new question with ID and timestamp
                        queries_t.insert([{
                            'id': 1,  # You might want to implement a proper ID system
                            'question': question,
                            'timestamp': datetime.now()
                        }])

                        # Fetch the latest answer
                        response = queries_t.select(queries_t.answer).order_by(
                            queries_t.timestamp, asc=False).limit(1).show()

                        if response:
                            st.write("Answer:", response[0]['answer'])
                        else:
                            st.error("No answer found. Please try again.")
                else:
                    st.warning("Please enter a question.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Clean up the temporary file
            os.unlink(tmp_file_path)


if __name__ == "__main__":
    main()
