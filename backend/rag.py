from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np

client = QdrantClient(host="localhost", port=6333)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

COLLECTION_NAME = "writing_context"

def embed_text(text):
    return embedding_model.encode(text).tolist()

def add_document_to_qdrant(doc_text, doc_id):
    vector = embed_text(doc_text)
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[{
            "id": doc_id,
            "vector": vector,
            "payload": {"text": doc_text, "section": "Introduction", "author_note": "Tone: speculative"}

        }]
    )

def retrieve_similar_context(query):
    query_vector = embed_text(query)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=3,
        filter=
        {"must": [{"key": "section", "match": {"value": "Introduction"}}]
    }
)
    return [hit.payload['text'] for hit in results]
