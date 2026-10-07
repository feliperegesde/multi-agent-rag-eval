# 🤖 Multi-Agent RAG with Semantic Routing & Automated Evaluation

Um pipeline end-to-end de **RAG Agêntico (Agentic RAG)** concebido com **LangGraph**, **FastAPI** e **Ragas**, integrando recuperação híbrida, auto-correção contra alucinações e esteira contínua de regressão de qualidade.

### 🌟 Destaques da Arquitetura
* **Roteamento Semântico:** Discriminação dinâmica entre consultas analíticas/metadados (Text-to-SQL seguro em SQLite) e consultas conceituais de documentação técnica.
* **Busca Híbrida & Re-ranking:** Fusão recíproca de posições (RRF) combinando BM25 e embeddings densos (`all-MiniLM-L6-v2`), refinados com Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
* **Ciclo de Auto-Correção:** Avaliador interno (*Critic Agent*) que detecta alucinações e reformula a query de busca em caso de falha de suporte factual.
* **LLMOps & CI/CD:** Avaliação contínua automatizada via GitHub Actions utilizando métricas Ragas (*faithfulness* e *answer relevancy*) sobre um *golden dataset*.
* **Serving de Baixa Latência:** API assíncrona em FastAPI com suporte a streaming de eventos de execução dos nós via SSE (*Server-Sent Events*).