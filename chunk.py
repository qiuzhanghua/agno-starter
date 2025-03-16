import os

from agno.agent import Agent
from agno.document.chunking.agentic import AgenticChunking

from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.pgvector import PgVector
from agno.models.ollama import Ollama
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.vectordb.qdrant import Qdrant
from agno.embedder.fastembed import FastEmbedEmbedder

api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")  #
qdrant_url = "http://localhost:6333"
collection_name = "thai-recipe-index"

qdrant_db = Qdrant(
    collection=collection_name,
    url=qdrant_url,
    api_key=api_key,
    embedder=FastEmbedEmbedder(384, "baai/bge-small-zh-v1.5"),
    # embedder=SentenceTransformerEmbedder(384, "all-minilm-l6-v2"),
    # all-mpnet-base-v2, all-minilm-l6-v2, baai/bge-large-zh-v1.5 ？
)


pdf_knowledge_base = PDFKnowledgeBase(
    path="/Users/q/Desktop",
    vector_db=qdrant_db,
    reader=PDFReader(chunk=True),
    chunking_strategy=AgenticChunking(model=Ollama(id="qwq")),
)

pdf_knowledge_base.load(recreate=True)

agent = Agent(
    model=Ollama(id="qwq"),
    knowledge=pdf_knowledge_base,
    search_knowledge=True,
)

# agent.print_response("How to make Thai curry?", markdown=True)

agent.print_response("课程表里有什么？", markdown=True)
