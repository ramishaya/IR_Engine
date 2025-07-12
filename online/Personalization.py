import pandas as pd
import sqlite3
import numpy as np  # ✅ ضروري
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def get_user_history(user_id):

    with sqlite3.connect('search_history.db') as conn:
        c = conn.cursor()
        c.execute("SELECT query FROM searches WHERE user_id = ?", (user_id,))
        queries = c.fetchall()
    return [q[0] for q in queries]

def build_user_profile(user_id, vectorizer):
    user_history = get_user_history(user_id)
    if not user_history:
        return None
    user_vector = vectorizer.transform(user_history)
    user_profile = user_vector.mean(axis=0)
    user_profile = np.asarray(user_profile)  # ✅ هذا هو الحل
    return user_profile

def get_personalized_recommendations(user_id, query, vectorizer, tfidf_matrix, alpha=0.7):
    print(f"[📌] تخصيص النتائج للمستخدم: {user_id}")
    user_profile = build_user_profile(user_id, vectorizer)
    query_vector = vectorizer.transform([query])

    if user_profile is None:
        similarity = cosine_similarity(query_vector, tfidf_matrix)
    else:
        sim_query = cosine_similarity(query_vector, tfidf_matrix)
        sim_profile = cosine_similarity(user_profile, tfidf_matrix)
        similarity = alpha * sim_query + (1 - alpha) * sim_profile

    recommendations = similarity.argsort()[0][-10:][::-1]
    scores = similarity[0, recommendations]

    return list(zip(recommendations, scores))
