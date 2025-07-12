import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

class HybridRetrieval:
    def __init__(self, dataset, tfidf_matrix, tfidf_vectorizer, w2v_model, doc_vectors):
        self.dataset = dataset
        self.tfidf_matrix = tfidf_matrix
        self.tfidf_vectorizer = tfidf_vectorizer
        self.w2v_model = w2v_model
        self.doc_vectors = doc_vectors

    def _clean_query_remotely(self, raw_query):
        try:
            response = requests.post("http://127.0.0.1:5001/clean_text", json={
                "dataset": self.dataset,
                "text": raw_query
            })
            if response.status_code != 200:
                print("❌ Failed to clean query from cleaning service.")
                return ""
            return response.json().get("cleaned_text", "")
        except Exception as e:
            print(f"❌ Exception in cleaning query: {e}")
            return ""

    def _embed_query(self, cleaned_query):
        tokens = cleaned_query.split()
        vectors = [self.w2v_model.wv[token] for token in tokens if token in self.w2v_model.wv]
        if not vectors:
            return np.zeros((1, self.doc_vectors.shape[1]))
        return np.mean(vectors, axis=0, keepdims=True)

    def _normalize(self, array):
        min_val = array.min()
        max_val = array.max()
        return (array - min_val) / (max_val - min_val + 1e-9)

    def retrieve(self, raw_query, alpha=0.7, prefilter_k=50, rerank_k=10):
        cleaned_query = self._clean_query_remotely(raw_query)
        if not cleaned_query:
            return []

        query_tfidf = self.tfidf_vectorizer.transform([cleaned_query])
        scores_tfidf = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores_tfidf)[-prefilter_k:]
        top_indices = [i for i in top_indices if i < len(self.doc_vectors)]
        if not top_indices:
            return []

        query_emb = self._embed_query(cleaned_query)
        sub_doc_emb = self.doc_vectors[top_indices]
        scores_w2v = cosine_similarity(query_emb, sub_doc_emb).flatten()

        norm_tfidf = self._normalize(scores_tfidf[top_indices])
        norm_w2v = self._normalize(scores_w2v)

        fused_scores = alpha * norm_tfidf + (1 - alpha) * norm_w2v
        ranked_order = np.argsort(fused_scores)[-rerank_k:][::-1]

        return [(int(top_indices[i]), float(fused_scores[i])) for i in ranked_order]

    def retrieve_top_only_w2v(self, raw_query, top_k=10):
        cleaned_query = self._clean_query_remotely(raw_query)
        if not cleaned_query:
            return []

        query_emb = self._embed_query(cleaned_query)
        if np.all(query_emb == 0):
            print("⚠️ الاستعلام لا يحتوي كلمات موجودة في Word2Vec.")
            return []

        scores = cosine_similarity(query_emb, self.doc_vectors).flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(i), float(scores[i])) for i in top_indices]
    

    
    def retrieve_with_personalization(self, query, user_profile, alpha=0.7, beta=0.3, prefilter_k=300, rerank_k=10):
        """
        استرجاع هجين مع تخصيص (بدمج استعلام المستخدم مع ملفه الشخصي)
        """
        q_tfidf = self.tfidf_vectorizer.transform([query])
        sim_tfidf = cosine_similarity(q_tfidf, self.tfidf_matrix).flatten()

        # word2vec
        tokens = query.split()
        w2v_vectors = [self.w2v_model.wv[t] for t in tokens if t in self.w2v_model.wv]
        sim_w2v = np.zeros_like(sim_tfidf)
        if w2v_vectors:
            q_w2v = np.mean(w2v_vectors, axis=0, keepdims=True)
            sim_w2v = cosine_similarity(q_w2v, self.doc_vectors).flatten()

        # ملف المستخدم (TF-IDF فقط)
        sim_user = np.zeros_like(sim_tfidf)
        if user_profile is not None:
            sim_user = cosine_similarity(user_profile, self.tfidf_matrix).flatten()

        sim_combined = alpha * sim_tfidf + (1 - alpha) * sim_w2v
        sim_final = beta * sim_combined + (1 - beta) * sim_user

        top_indices = sim_final.argsort()[-rerank_k:][::-1]
        scores = sim_final[top_indices]
        return list(zip(top_indices, scores))
