from pixeltable.iterators.document import DocumentSplitter
from pixeltable.functions.huggingface import sentence_transformer
import pixeltable as pxt
import numpy as np
import urllib.request

# Create the Pixeltable workspace
pxt.create_dir('rag_demo1', ignore_errors=True)

# Make sure we start with a clean slate
pxt.drop_table('rag_demo1.chunks', ignore_errors=True)
pxt.drop_table('rag_demo1.documents', ignore_errors=True)
pxt.drop_table('rag_demo1.queries', ignore_errors=True, force=True)
pxt.drop_table('rag_demo1.docs', ignore_errors=True)

# Create a table to store documents
docs = pxt.create_table('rag_demo1.docs', {
    'source_doc': pxt.DocumentType()
})

# Download the spreadsheet
base = 'https://github.com/pixeltable/pixeltable/raw/master/docs/source/data/rag-demo/'
qa_url = base + 'Q-A-Rag.xlsx'
qa_filename, _ = urllib.request.urlretrieve(qa_url)

# Create a table from the spreadsheet
queries_t = pxt.io.import_excel('rag_demo1.queries', qa_filename)
documents_t = pxt.create_table(
    'rag_demo1.documents', {'document': pxt.DocumentType()}
)

# Local file path of the single PDF file
file_path = r"D:\Pixeltable-AI-Projects\Pixeltable-1-GPT-RAG\PO.pdf"

# Insert the single PDF document into the table
try:
    documents_t.insert([{'document': file_path}])
except pxt.exceptions.Error as e:
    print(f"Error inserting document: {e}")

# Create a view to split documents into chunks
chunks_t = pxt.create_view(
    'rag_demo1.chunks',
    documents_t,
    iterator=DocumentSplitter.create(
        document=documents_t.document,
        separators='token_limit', limit=300
    )
)


@pxt.expr_udf
def e5_embed(text: str) -> np.ndarray:
    return sentence_transformer(text, model_id='intfloat/e5-large-v2')


# Explicitly register the function path
e5_embed.self_path = 'e5_embed'

# Add embedding index to the chunks table
chunks_t.add_embedding_index('text', text_embed=e5_embed)

while True:
    query_text = input("User> ")
    if query_text.lower() == "exit":
        break
    else:
        sim = chunks_t.text.similarity(query_text)
        nvidia_eps_query = (
            chunks_t.order_by(sim, asc=False)
            .select(similarity=sim, text=chunks_t.text)
            .limit(5)
        )
        print(nvidia_eps_query.collect())
