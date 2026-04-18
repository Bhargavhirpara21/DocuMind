from __future__ import annotations

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from src import config


def get_vector_store() -> ChromaVectorStore:
    config.ensure_dirs()
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_or_create_collection(config.CHROMA_COLLECTION)
    return ChromaVectorStore(chroma_collection=collection)


def build_index(chunks: list[TextNode], embed_model) -> VectorStoreIndex:
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(
        nodes=chunks, storage_context=storage_context, embed_model=embed_model
    )


def load_index(embed_model) -> VectorStoreIndex:
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, embed_model=embed_model
    )
