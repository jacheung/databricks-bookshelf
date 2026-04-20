# RAG Blueprints 🧺

Three blueprints for building Retrieval-Augmented Generation (RAG) systems on Databricks. They share a common parameterized config pattern but differ in vector store, deployment target, and complexity.

---

## How RAG Works

```
                        ┌─────────────────────────────────────────────┐
                        │                                             │
          (1) query     │   ┌─────────────────────────────────────┐   │
  ┌──────┐────────────► │   │              LLM                    │   │
  │ User │             │   │         (e.g. llama-3, gpt-4)       │   │
  └──────┘◄────────────│   └──────────────┬──────────────────────┘   │
        (6) answer      │                 │  (2) embed query          │
                        │                 ▼                           │
                        │   ┌─────────────────────────────────────┐   │
                        │   │         Embedding Model             │   │
                        │   └──────────────┬──────────────────────┘   │
                        │                 │  (3) query vector         │
                        │                 ▼                           │
                        │   ┌─────────────────────────────────────┐   │
                        │   │         Vector Database             │   │
                        │   │   (Databricks VS / FAISS / etc.)    │   │
                        │   └──────────────┬──────────────────────┘   │
                        │                 │  (4) top-k doc chunks     │
                        │                 ▼                           │
                        │   ┌─────────────────────────────────────┐   │
                        │   │     LLM + retrieved context         │   │
                        │   │   "Answer using only these docs"    │   │
                        │   └─────────────────────────────────────┘   │
                        │                                   (5) grounded response
                        └─────────────────────────────────────────────┘
```

**The key idea:** instead of relying purely on the LLM's training data, the query is first converted into an embedding vector and used to retrieve the most relevant document chunks from a vector database. Those chunks are injected into the prompt as context, so the LLM's answer is grounded in your data rather than general knowledge.

---

## [parameterized-rag-chain](./parameterized-rag-chain)

**The baseline.** A RAG chatbot powered by **Databricks Vector Search** and a Databricks-hosted LLM endpoint. Everything — catalog, table names, model names — is driven by a single `rag_chain_config.yaml`. Three notebooks handle index creation, chain definition, and deployment (with a Review App attached for human feedback).

Best for: teams already on the Databricks stack who want a clean, production-ready starting point with MLflow logging and a review UI out of the box.

---

## [OSS-parameterized-rag-chain](./OSS-parameterized-rag-chain)

**The portable variant.** Swaps Databricks Vector Search for a **FAISS** vector database stored in DBFS. Also adds a dedicated notebook for deploying a completions model with **provisioned throughput**. The vector DB is bundled directly into the MLflow model artifacts at log time, so inference is self-contained.

Best for: use cases where you want a fully open-source vector store, need to control throughput tightly, or want the index baked into the model artifact rather than queried at runtime.

---

## [parameterized-rag-agent](./parameterized-rag-agent)

**The agent upgrade.** A single-file (`rag-agent.py`) implementation that wraps a RAG chain inside a **Databricks agent** framework. Compared to the chain blueprints, this adds agent-level orchestration and is structured for tool-calling patterns rather than a simple chain.

Best for: scenarios where the retrieval step is one tool among many, or where you want to extend the RAG pattern toward multi-step reasoning.

---

## Quick comparison

| | parameterized-rag-chain | OSS-parameterized-rag-chain | parameterized-rag-agent |
|---|---|---|---|
| Vector store | Databricks Vector Search | FAISS (DBFS) | Databricks Vector Search |
| LLM serving | Databricks endpoint | Provisioned throughput endpoint | Databricks endpoint |
| Structure | 3 notebooks | 3 notebooks | 1 file |
| Review App | Yes | No | No |
| Pattern | Chain | Chain | Agent |
