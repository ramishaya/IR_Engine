from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
from joblib import load
from scipy.sparse import load_npz
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os
import requests
import sqlite3
from Query_Refinement import correct_spelling, suggest_queries
# from HybridRetrieval import HybridRetrieval
# from HybridRetrieval import retrieve_with_personalization
# retrieve_with_personalization
from Personalization import get_personalized_recommendations
from Personalization import build_user_profile
# تحميل أدوات NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__, static_folder='static', template_folder='templates')
cached_resources = {}

# تحميل الموارد


OFFLINE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../offline'))

def load_all_resources():
    print("📦 تحميل الموارد...")

    # Quora
    cached_resources['vectorizer_quora'] = load(os.path.join(OFFLINE_PATH, 'vectorizer_quora.joblib'))
    cached_resources['tfidf_quora'] = load_npz(os.path.join(OFFLINE_PATH, 'tfidf_quora.npz'))
    cached_resources['data_quora'] = pd.read_csv(os.path.join(OFFLINE_PATH, "cleaned_quora.csv")).dropna(subset=["cleaned_text"]).reset_index(drop=True)
    cached_resources['w2v_quora'] = Word2Vec.load(os.path.join(OFFLINE_PATH, "word2vec_quora.model"))
    cached_resources['doc_vecs_quora'] = load(os.path.join(OFFLINE_PATH, 'quora_doc_vectors.joblib'))

    # Antique
    cached_resources['vectorizer_antique'] = load(os.path.join(OFFLINE_PATH, 'vectorizer_antique.joblib'))
    cached_resources['tfidf_antique'] = load_npz(os.path.join(OFFLINE_PATH, 'tfidf_antique.npz'))
    cached_resources['data_antique'] = pd.read_csv(os.path.join(OFFLINE_PATH, "antique.csv"))
    cached_resources['w2v_antique'] = Word2Vec.load(os.path.join(OFFLINE_PATH, "word2vec_antique_200.model"))
    cached_resources['doc_vecs_antique'] = load(os.path.join(OFFLINE_PATH, 'antique_doc_vectors_2000.joblib'))

    print("✅ تم تحميل الموارد.")


load_all_resources()

# API بوست مان مع personalization
@app.route('/search', methods=['GET'])
def search():
    raw_query = request.args.get('query', "")
    dataset = request.args.get('dataset', "quora").lower()
    representation = request.args.get('representation', "tfidf").lower()
    use_personalization = request.args.get('use_personalization', 'false').lower() == 'true'
    user_id = request.args.get('user_id', None)

    # ✅ تحقق من الإدخال
    if not raw_query:
        return jsonify({"error": "Query is required"}), 400
    if dataset not in ['quora', 'antique']:
        return jsonify({"error": "Invalid dataset"}), 400
    if representation not in ['tfidf', 'word2vec', 'hybrid']:
        return jsonify({"error": "Invalid representation"}), 400
    if use_personalization and not user_id:
        return jsonify({"error": "User ID is required for personalization"}), 400

    # ✅ تصحيح وتصفية الاستعلام
    corrected_query = correct_spelling(raw_query)
    try:
        response = requests.post("http://127.0.0.1:5001/clean_text", json={"dataset": dataset, "text": corrected_query})
        if response.status_code != 200:
            return jsonify({"error": "Text cleaning service failed"}), 500
        cleaned_query = response.json().get("cleaned_text", "")
    except Exception as e:
        return jsonify({"error": f"Cleaning service error: {str(e)}"}), 500

    # ✅ تحميل الموارد
    vec = cached_resources[f'vectorizer_{dataset}']
    tfidf = cached_resources[f'tfidf_{dataset}']
    data = cached_resources[f'data_{dataset}']
    w2v = cached_resources[f'w2v_{dataset}']
    doc_vecs = cached_resources[f'doc_vecs_{dataset}']
    text_col = 'text'

    results = []

    # ✅ حالة التمثيل الهجين
    if representation == 'hybrid':
        hybrid = HybridRetrieval(dataset, tfidf, vec, w2v, doc_vecs)

        if use_personalization:
            from Personalization import build_user_profile
            user_profile = build_user_profile(user_id, vec)
            hybrid_results = hybrid.retrieve_with_personalization(cleaned_query, user_profile, alpha=0.7, beta=0.3)
        else:
            hybrid_results = hybrid.retrieve(cleaned_query, alpha=0.8)

        results = [{
            "rank": i + 1,
            "doc_id": str(data.iloc[doc_id]["doc_id"]),
            "score": round(score, 4),
            "text": str(data.iloc[doc_id][text_col])[:1000].replace("\n", " ")
        } for i, (doc_id, score) in enumerate(hybrid_results)]

    # ✅ حالة tfidf أو word2vec
    else:
        if use_personalization:
            from Personalization import get_personalized_recommendations
            personalized = get_personalized_recommendations(user_id, cleaned_query, vec, tfidf)
            if personalized:
                top_indices, top_scores = zip(*personalized)
            else:
                q_vec = vec.transform([cleaned_query])
                sims = cosine_similarity(q_vec, tfidf).flatten()
                top_indices = sims.argsort()[-10:][::-1]
                top_scores = sims[top_indices]
        elif representation == 'tfidf':
            q_vec = vec.transform([cleaned_query])
            sims = cosine_similarity(q_vec, tfidf).flatten()
            top_indices = sims.argsort()[-10:][::-1]
            top_scores = sims[top_indices]
        elif representation == 'word2vec':
            tokens = cleaned_query.split()
            vectors = [w2v.wv[t] for t in tokens if t in w2v.wv]
            if not vectors:
                return jsonify({"error": "No known word2vec tokens in query."}), 400
            q_vec = np.mean(vectors, axis=0, keepdims=True)
            sims = cosine_similarity(q_vec, doc_vecs).flatten()
            top_indices = sims.argsort()[-10:][::-1]
            top_scores = sims[top_indices]

        results = [{
            "rank": i + 1,
            "doc_id": str(data.iloc[idx]["doc_id"]),
            "score": round(score, 4),
            "text": str(data.iloc[idx][text_col])[:1000].replace("\n", " ")
        } for i, (idx, score) in enumerate(zip(top_indices, top_scores))]

    # ✅ حفظ الاستعلام في قاعدة البيانات
    if user_id:
        with sqlite3.connect('search_history.db') as conn:
            c = conn.cursor()
            c.execute("INSERT INTO searches (user_id, query, dataset) VALUES (?, ?, ?)", (user_id, raw_query, dataset))
            conn.commit()

    # ✅ إخراج النتائج
    return jsonify({
        "dataset": dataset,
        "representation": representation,
        "use_personalization": use_personalization,
        "query": raw_query,
        "corrected_query": corrected_query,
        "cleaned_query": cleaned_query,
        "results": results
    })

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/correct', methods=['GET'])
def correct():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400
    corrected_query = correct_spelling(query)
    return jsonify({"corrected_query": corrected_query})

@app.route('/suggest', methods=['GET'])
def suggest():
    dataset = request.args.get('dataset')
    query = request.args.get('query')
    if not query or not dataset:
        return jsonify([])

    suggestions = set()

    with sqlite3.connect('search_history.db') as conn:
        c = conn.cursor()
        c.execute("SELECT query FROM searches WHERE query LIKE ?", ('%' + query + '%',))
        db_suggestions = c.fetchall()
        for s in db_suggestions:
            suggestions.add(s[0])

    if dataset == 'quora':
        vec = cached_resources.get('vectorizer_quora_for_refinment')
    elif dataset == 'antique':
        vec = cached_resources.get('vectorizer_antique_for_refinment')
    else:
        return jsonify({"error": "Invalid dataset"}), 400

    if vec:
        suggestions.update(suggest_queries(query, vec))

    return jsonify(list(suggestions))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
