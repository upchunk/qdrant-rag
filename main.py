import logging
import sys

import qdrant_client
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

load_dotenv()

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def main():
    client = qdrant_client.QdrantClient(location=":memory:")

    documents = SimpleDirectoryReader("./data").load_data()

    vector_store = QdrantVectorStore(client=client, collection_name="paul_graham")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    llm = OpenAI(model="gpt-4.1-mini", max_tokens=16000)

    chat_engine = index.as_chat_engine(llm=llm)

    response = chat_engine.chat("is the summary of the document")
    print(response.response)


if __name__ == "__main__":
    main()
