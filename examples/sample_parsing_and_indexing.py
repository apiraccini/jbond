from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FILES_FOLDER = PROJECT_ROOT / "data"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25" # or "prithivida/Splade_PP_en_v1"
CHUNK_MAX_TOKENS = 512
COLLECTION_NAME = "test_collection"

def main():

    # Parse files
    parser = DocumentConverter()
    parsed_files = parser.convert_all(INPUT_FILES_FOLDER.iterdir())

    # Obtain chunks
    chunker = HybridChunker(tokenizer=EMBEDDING_MODEL, max_tokens=CHUNK_MAX_TOKENS)
    chunks = []
    for file in parsed_files:
        doc_chunks = list(chunker.chunk(file.document))
        chunks.extend(doc_chunks)

    # Index chunks
    db_client = QdrantClient(":memory:") # or QdrantClient(path=PROJECT_ROOT / "index.db")
    db_client.set_model(EMBEDDING_MODEL)
    db_client.set_sparse_model(SPARSE_MODEL) # comment to use only dense vector search

    documents, metadata = [], []
    for chunk in chunks:
        documents.append(chunk.text)
        metadata.append(chunk.meta.export_json_dict())
    
    db_client.add(collection_name=COLLECTION_NAME, documents=documents, metadata=metadata)

    # Sample query
    query = "What is the somatosensory system?"
    results = db_client.query(collection_name=COLLECTION_NAME, query_text=query, limit=3)

    print(f"Structure of retrieved documents: {dir(results[0])}")
    print("Retrieved documents:")
    for result in results:
        print(result.document)

if __name__ == "__main__":
    main()