import os
import time
import re
import glob
import networkx as nx
from fastapi import FastAPI, Request
from transformers import pipeline, AutoTokenizer
from prometheus_client import start_http_server, Summary, Counter, Gauge
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
import nltk

# =========================
# Configuration
# =========================
DATA_DIR = "data"
PROM_PORT = 8001

# =========================
# Metrics setup
# =========================
qa_latency = Summary('graphrag_latency_seconds', 'Latency of GraphRAG QA')
qa_requests = Counter('graphrag_requests_total', 'Number of QA requests')
graph_nodes = Gauge('graphrag_nodes_total', 'Number of graph nodes')
graph_edges = Gauge('graphrag_edges_total', 'Number of graph edges')

app = FastAPI()

# =========================
# Load multiple text files and build graph
# =========================
print("Loading text files and building knowledge graph...")
nltk.download('punkt')

ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
G = nx.Graph()
texts = []

for filepath in glob.glob(os.path.join(DATA_DIR, "*.txt")):
    filename = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    texts.append((filename, content))
    sentences = nltk.sent_tokenize(content)

    for sent in sentences:
        entities = ner_pipeline(sent)
        ents = [e['word'].replace("##", "").strip() for e in entities]
        for e in ents:
            G.add_node(e, label="ENTITY", source_file=filename)
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                G.add_edge(ents[i], ents[j], relation="co-occurs", source_file=filename)

graph_nodes.set(len(G.nodes))
graph_edges.set(len(G.edges))
print(f"Graph built with {len(G.nodes)} nodes and {len(G.edges)} edges across {len(texts)} files")

# =========================
# Embedding and LLM setup
# =========================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Loading local language model...")
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device_map="auto",
    max_new_tokens=256,
    temperature=0.2,
)

qa_chain = LLMChain(
    llm=HuggingFacePipeline(pipeline=hf_pipeline),
    prompt=PromptTemplate.from_template("Context:\n{context}\n\nQuestion: {q}\nAnswer:")
)

# =========================
# Build FAISS vector index
# =========================
print("Building FAISS vector index...")
nodes = [
    Document(page_content=n, metadata={"label": G.nodes[n]["label"], "source": G.nodes[n].get("source_file", "unknown")})
    for n in G.nodes
]
db = FAISS.from_documents(nodes, embeddings)
retriever = db.as_retriever()

# =========================
# Helper functions
# =========================
def clean_bert_subwords(text: str) -> str:
    text = re.sub(r"##", "", text)
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def subgraph_context(query):
    results = retriever.get_relevant_documents(query)
    node_names = [r.page_content for r in results]
    neighbors = set()
    for n in node_names:
        neighbors.update(G.neighbors(n))
    subgraph = G.subgraph(list(node_names) + list(neighbors))
    context = " ".join(subgraph.nodes)
    return context, subgraph

def run_query(q: str):
    #qa_requests.inc()
    start_time = time.time()

    # Group results per file
    relevant_docs = retriever.get_relevant_documents(q)
    files = {}
    for doc in relevant_docs:
        src = doc.metadata.get("source", "unknown")
        files.setdefault(src, []).append(doc.page_content)

    responses = {}
    for src, nodes in files.items():
        context = " ".join(nodes)
        raw_answer = qa_chain.run({"context": context, "q": q})
        cleaned_answer = clean_bert_subwords(raw_answer)
        responses[src] = cleaned_answer

    latency = time.time() - start_time
    print(f"Query latency: {latency:.2f}s | Files used: {list(responses.keys())}")

    return {
        "query": q,
        "answers_by_file": responses,
        "files_consulted": list(responses.keys()),
        "latency_sec": latency
    }

# =========================
# FastAPI endpoints
# =========================
@app.get("/query")
@qa_latency.time()
def query_get(q: str = None):
    if not q:
        return {"error": "Please provide a query parameter 'q' in the URL."}
    qa_requests.inc()  # Increment only for valid queries
    return run_query(q)

@app.post("/query")
@qa_latency.time()
def query_post(payload: dict):
    q = payload.get("question")
    if not q:
        return {"error": "Please provide a 'question' field in JSON body."}
    qa_requests.inc()  # Increment only for valid queries
    return run_query(q)

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    print(f"Starting GraphRAG evaluator (Prometheus on port {PROM_PORT})")
    start_http_server(PROM_PORT)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
