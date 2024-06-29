import urllib
from pixeltable.iterators.document import DocumentSplitter
from pixeltable.functions.huggingface import sentence_transformer
import pixeltable as pxt
import numpy as np
import urllib.request
import tiktoken
import logging

# Configure logging to use UTF-8
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    encoding='utf-8')

# Make sure we start with a clean slate
pxt.drop_table('rag_demo2.chunks', ignore_errors=True)
pxt.drop_table('rag_demo2.documents', ignore_errors=True)
pxt.drop_table('rag_demo2.queries', ignore_errors=True, force=True)
pxt.drop_table('rag_demo2.docs', ignore_errors=True)

pxt.create_dir('rag_demo2', ignore_errors=True)
# pxt.create_dir('demo', ignore_errors=True)


# t = pxt.create_table('demo.openai', {'id': pxt.IntType(), 'input': pxt.StringType()})
pxt.drop_table('rag_demo2.docs', ignore_errors=True)
docs = pxt.create_table('rag_demo2.docs', {
    'source_doc': pxt.DocumentType()
})


# Local file path of the single PDF file
file_path = r"D:\Pixeltable-AI-Projects\Pixeltable-1-GPT-RAG\PO.pdf"

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

# chunks_t['new'] =sentence_transformer(chunks_t.text, model_id='paraphrase-MiniLM-L6-v2')

# chunks_t.head(1)


while True:
    query_text = input("User>: ")
    if query_text.lower() == 'exit':
        print("Thanks!")
        break
    else:
        sim = chunks_t.text.similarity(query_text)
        ai_response = (
            chunks_t.order_by(sim, asc=False)
            .select(similarity=sim, text=chunks_t.text)
            .limit(1)
        )
        final_res = ai_response.collect()
        print(f"Assistabt> {final_res}")
