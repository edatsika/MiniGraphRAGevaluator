# 🧩 MiniGraphRAG Evaluator

A lightweight **Graph-based Retrieval-Augmented Generation (GraphRAG)** evaluator built with **FastAPI**, **LangChain**, **Prometheus**, and **Grafana**.  
This project demonstrates how to analyze and query textual data using graph relationships and monitor model performance metrics in real time.

---

## 🚀 Features

- 🧩 **Graph Construction:** Automatically extracts entities and relationships from text files using BERT-NER.  
- 🔎 **Semantic Search:** Uses sentence embeddings with FAISS for context retrieval.  
- 🗣️ **Question Answering:** Generates answers via a local `Flan-T5` model (no external API).  
- 📊 **Metrics Monitoring:** Tracks query latency, graph size, and request counts with Prometheus.  
- 📈 **Dashboard Visualization:** Displays real-time performance in Grafana.  
- 📁 **Multi-file Support:** Easily extend to multiple `.txt` files for broader context.

---

## 🐳 Quick start

### Clone the repository, build and start the services and access the components
```bash
git clone https://github.com/edatsika/MiniGraphRAGevaluator.git
cd MiniGraphRAGevaluator
docker compose up --build
```

| Service          | URL                                                      | Description              |
| ---------------- | -------------------------------------------------------- | ------------------------ |
| **GraphRAG API** | [http://localhost:8100/docs](http://localhost:8100/docs) | FastAPI interactive docs |
| **Prometheus**   | [http://localhost:9190](http://localhost:9190)           | Metrics explorer         |
| **Grafana**      | [http://localhost:3100](http://localhost:3100)           | Dashboard visualization  |

---
### 🌐 **API Examples**
🔹 GET Request
```bash
curl "http://localhost:8100/query?q=What is the main university in Ioannina?"
```
🔹 POST Request
```bash
curl -X POST "http://localhost:8100/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "Who is the mayor of Ioannina?"}'
```     
### ✅ Example Response
```bash
{
  "query": "Who is the mayor of Ioannina?",
  "answers_by_file": {
    "data/ioannina1.txt": "The mayor of Ioannina is Moses Elisaf."
  },
  "files_consulted": ["data/ioannina1.txt"],
  "latency's": 3.41
}
```
## 📊 Monitoring
🔸 Prometheus
```bash
http://localhost:9190/targets
Username: admin
Password: admin
```
Add Prometheus data source:
```bash
URL: http://prometheus2:9090 (or http://localhost:9190)
```
| Metric                     | Description                            |
| -------------------------- | -------------------------------------- |
| `graphrag_requests_total`  | Number of valid user queries processed |
| `graphrag_latency_seconds` | Query latency in seconds               |
| `graphrag_nodes_total`     | Number of graph nodes (entities)       |
| `graphrag_edges_total`     | Number of graph edges (relations)      |

## 📚 Adding Your Own Data

Add .txt files into the data/ directory, for example:
```bash
data/
├── sample1.txt
├── sample2.txt
```
Each file is processed separately, and answers are grouped by source file.

## 🧩 Stack
| Component        | Technology                               |
| ---------------- | ---------------------------------------- |
| API              | FastAPI                                  |
| NER              | `dslim/bert-base-NER`                    |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM              | `google/flan-t5-base`                    |
| Vector Search    | FAISS                                    |
| Graph Engine     | NetworkX                                 |
| Monitoring       | Prometheus + Grafana                     |
| Containerization | Docker Compose                           |

## 💡 Example Grafana Panels
- Total queries over time
- Average query latency
- Graph node/edge counts
- Requests per second
- Active file sources

## 📘 How It Works

- The system loads all .txt files from the /data directory.
- Each file is split into sentences using NLTK.
- Entities are extracted via BERT NER and used to build a graph (networkx).
- Sentence embeddings are computed using sentence-transformers/all-MiniLM-L6-v2.
- A FAISS vector store enables similarity search for query contexts.
- A Flan-T5 model answers user queries based on retrieved context.
- All requests, latency, and graph stats are exposed as Prometheus metrics.
