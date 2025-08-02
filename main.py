import logging
import sys
from functools import partial
from typing import Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from ftfy import fix_text
from llama_index.core import (
    Settings,
    VectorStoreIndex,
)
from llama_index.core.agent import FunctionAgent, ToolCallResult
from llama_index.core.schema import Document, MediaResource, NodeWithScore
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.tools import RetrieverTool
from llama_index.core.workflow import Context
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel
from pypdf import PdfReader
from qdrant_client import AsyncQdrantClient, QdrantClient
from tiktoken import get_encoding

# Load Local .env file, where you store the
load_dotenv()

# Initiate Logger for Debugging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(stream=sys.stdout))


# Initialize FastAPI App
app = FastAPI(title="RAG System with QDrant Vector Database")

# Enable CORS for "*", Make Endpoint Easily Accessible
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize Global Embedding Model
Settings.embed_model = OpenAIEmbedding(
    model=OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL,
)

# Initialzie Global LLM
Settings.llm = OpenAI(model="gpt-4.1-mini", temperature=0.1)

# Initialize Global Tokenizer
Settings.tokenizer = partial(get_encoding("o200k_base").encode, allowed_special="all")

# Initilize In Memory Qdrant Client -> Will store data in memory
QDRANT_CLIENT = QdrantClient(location=":memory:")
ASYNC_QDRANT_CLIENT = AsyncQdrantClient(location=":memory:")


def get_token_count(string: str):
    tokenizer = Settings.tokenizer
    return len(tokenizer(string))


def to_snake(string: str):
    return "_".join(string.lower().split())


@app.post("/index-document")
async def index_file_to_qdrant(file: UploadFile):
    vector_store = QdrantVectorStore(
        aclient=ASYNC_QDRANT_CLIENT,
        collection_name="document_index",
    )
    indexed_node_id: list[str] = []

    # Read PDF File
    with PdfReader(file.file) as pdf:
        pdf_chunk_sizes: list[int] = []
        text_lists: list[str] = []

        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "page_count": pdf.get_num_pages(),
        }

        for page in pdf.pages:
            # Extract Text on the page
            text = page.extract_text(extraction_mode="layout")
            text = fix_text(text)  # Fix in case there is any parsing Error

            # Simple Spacing Cleanup:
            text = " ".join([t for t in text.split(" ") if t])
            print({page.page_number: text})

            # Calculate Page Token Count as Chunk Size
            page_chunk_size = get_token_count(text)

            text_lists.append(text)
            pdf_chunk_sizes.append(page_chunk_size)

        # For Indexing based on Document
        combined_text = "\n\n\n".join(text_lists)

        # Initialize Sentence Splitter
        ## Set Chunk Size based page chunk size
        avg_chunk_size = sum(pdf_chunk_sizes) // len(pdf_chunk_sizes)
        sentence_splitter = SentenceSplitter.from_defaults(
            chunk_size=avg_chunk_size, chunk_overlap=avg_chunk_size // 8
        )

        # Initialize Document Node
        document = Document(
            id_=to_snake(file.filename),
            text_resource=MediaResource(text=combined_text, mimetype=file.content_type),
            metadata=metadata,
        )

        # Split Document into Chunks
        nodes = await sentence_splitter.aget_nodes_from_documents([document])
        for num, node in enumerate(nodes, 1):
            node_text = node.get_content()
            node.metadata = {
                **metadata,
                "split": f"{num} of {len(nodes)}",
            }  # Set Split Metadata, Help Knowing Splits of Document

            # Get Chunk Embedding
            node.embedding = await Settings.embed_model.aget_text_embedding(node_text)

        # Index Chunk Node into Vector Store
        indexed_node_id = await vector_store.async_add(nodes)

    if indexed_node_id:
        return JSONResponse(
            {
                "msg": "Document Succesfully Indexed",
                "node_ids": indexed_node_id,
                **metadata,
            }
        )
    return JSONResponse(
        {
            "msg": "Document Indexing Failed",
            "node_ids": indexed_node_id,
            **metadata,
        },
        status_code=status.HTTP_400_BAD_REQUEST,
    )


@app.post("/index-document-pages")
async def index_doc_pages_to_qdrant(file: UploadFile):
    vector_store = QdrantVectorStore(
        aclient=ASYNC_QDRANT_CLIENT,
        collection_name="document_pages_index",
    )
    indexed_node_id: list[str] = []

    # Read PDF and Parse PDF File
    with PdfReader(file.file) as pdf:
        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
        }

        for page in pdf.pages:
            text = page.extract_text(extraction_mode="layout")
            page_metadata = {**metadata, "page": page.page_number}
            text = fix_text(text)  # Fix In Case there is any parsing Error

            # Simple Spacing Cleanup:
            text = " ".join([t for t in text.split(" ") if t])
            print({page.page_number: text})

            # Calculate Page Token Count
            page_chunk_size = get_token_count(text)

            # Set Chunk size to reasonable sizes, trying to capture page sections
            chunk_size = page_chunk_size // 2
            sentence_splitter = SentenceSplitter.from_defaults(
                chunk_size=chunk_size, chunk_overlap=chunk_size // 4
            )

            # Initialize Document Page Node
            document = Document(
                id_=to_snake(file.filename),
                text_resource=MediaResource(text=text, mimetype=file.content_type),
                metadata=page_metadata,
            )

            # Split Document into Chunks
            nodes = await sentence_splitter.aget_nodes_from_documents([document])
            for num, node in enumerate(nodes, 1):
                node_text = node.get_content()
                node.metadata = {**page_metadata, "split": f"{num} of {len(nodes)}"}

                # Get Chunk Embedding
                node.embedding = await Settings.embed_model.aget_text_embedding(
                    node_text
                )

            node_id = await vector_store.async_add(nodes)
            indexed_node_id.append(node_id)

    if indexed_node_id:
        return JSONResponse(
            {
                "msg": "Document Succesfully Indexed",
                "node_ids": indexed_node_id,
                **metadata,
            }
        )
    return JSONResponse(
        {
            "msg": "Document Indexing Failed",
            "node_ids": indexed_node_id,
            **metadata,
        },
        status_code=status.HTTP_400_BAD_REQUEST,
    )


class RAGRequest(BaseModel):
    question: str
    retrieve_from: Literal["document_index", "document_pages_index"] = "document_index"


# Setup Agent Prompt based on underlying LLM. set Task, Role, Steps, Rules, and Expected Output
# Reference for LLM used in this program: https://cookbook.openai.com/examples/gpt4-1_prompting_guide
AGENT_PROMPT = """
You are an AI Agent, specialized in answering user questions accurately with precission
Your task is to answer user questions using the information provided only from the available tools

Step:
1. Analyze user question. if it's unclear, ask user to clarify their question
2. using the `retriever_tool`, retrieve relevant documents / document chunks using semantic keywords or natural language query
3. Analyze the retrieved documents / chunks:
   - If the documents / chunks is adequate to answer the question, continue to the next step
   - If the documents / chunks doesn't match with the question or inadequate, retry the retrieval process with different keywords or query
4. Answer the question based on the information provided on the documents / chunks. give your reasoning too

Expected Output:

```
Answer: "string"
Reason: "string"
```
"""


@app.post("/document-rag")
async def document_rag(q: RAGRequest):
    vector_store = QdrantVectorStore(
        aclient=ASYNC_QDRANT_CLIENT,
        collection_name=q.retrieve_from,
    )
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Conditional Top K, Adjust Based on Need
    top_k = 2 if q.retrieve_from == "document_index" else 5

    # Initialize Retriever
    retriever = index.as_retriever(similarity_top_k=top_k)
    agent_tools = [
        RetrieverTool.from_defaults(
            retriever=retriever,
            name="retriever_tool",
            description="Useful for running a natural language query against a knowledge base and retrieving a set of relevant documents.",
        )
    ]

    # Initialize Function Agent
    agent = FunctionAgent(system_prompt=AGENT_PROMPT, tools=agent_tools)
    ctx = Context(agent)

    # Initialzie Ageent Run Handler
    handler = agent.run(user_msg=q.question, ctx=ctx)

    reference = []
    async for ev in handler.stream_events():
        # Check for Reference
        if isinstance(ev, ToolCallResult) and ev.tool_output.raw_output:
            tool_output = ev.tool_output.raw_output
            if isinstance(tool_output, list):
                for each in tool_output:
                    if isinstance(each, NodeWithScore):
                        reference.append(
                            each.model_dump(
                                mode="json",
                                exclude_defaults=True,
                                exclude_none=True,
                                exclude_unset=True,
                            )
                        )

    # Check Agent Final Answer
    response = await handler

    return JSONResponse(
        {"question": q.question, "answer": str(response), "reference": reference}
    )


@app.get("/")
def home():
    return RedirectResponse(url="/docs")  # Redirect to OpenAPI Swagger Docs


def main():
    uvicorn.run("main:app", host="0.0.0.0")


if __name__ == "__main__":
    main()
