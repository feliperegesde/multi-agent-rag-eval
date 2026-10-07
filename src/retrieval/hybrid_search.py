import os
import glob
from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

class HybridRetriever:
    def __init__(self, docs_folder: str = "data/docs"):
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        self.chunks: List[Dict[str, Any]] = []
        self._load_and_chunk_documents(docs_folder)
        
        # Inicia BM25
        tokenized_corpus = [doc["text"].lower().split(" ") for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Inicia Embeddings Densos
        texts = [doc["text"] for doc in self.chunks]
        self.dense_embeddings = self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def _load_and_chunk_documents(self, docs_folder: str, chunk_size: int = 150):
        md_files = glob.glob(f"{docs_folder}/*.md")
        for filepath in md_files:
            filename = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                words = content.split()
                for i in range(0, len(words), chunk_size):
                    chunk_text = " ".join(words[i:i + chunk_size])
                    if len(chunk_text.strip()) > 50:
                        self.chunks.append({
                            "source": filename,
                            "text": chunk_text
                        })

    def search(self, query: str, top_k: int = 4, rrf_k: int = 60) -> List[Dict[str, Any]]:
        # 1. BM25 Search
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked = np.argsort(bm25_scores)[::-1]
        
        # 2. Dense Search
        query_embedding = self.encoder.encode([query], normalize_embeddings=True)[0]
        dense_scores = np.dot(self.dense_embeddings, query_embedding)
        dense_ranked = np.argsort(dense_scores)[::-1]
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, idx in enumerate(bm25_ranked[:30]):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))
            
        for rank, idx in enumerate(dense_ranked[:30]):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))
            
        candidates_idx = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:10]
        candidate_chunks = [self.chunks[idx] for idx in candidates_idx]
        
        # 4. Cross-Encoder Re-Ranking
        pairs = [[query, c["text"]] for c in candidate_chunks]
        rerank_scores = self.reranker.predict(pairs)
        
        for c, score in zip(candidate_chunks, rerank_scores):
            c["score"] = float(score)
            
        candidate_chunks = sorted(candidate_chunks, key=lambda x: x["score"], reverse=True)
        return candidate_chunks[:top_k]