import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import joblib

df = pd.read_csv("cleaned_quora.csv").dropna(subset=["cleaned_text"]).reset_index(drop=True)

model = Word2Vec.load("word2vec_quora.model")

def get_avg_vector(text, model):
    words = text.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

vectors = np.vstack(df['cleaned_text'].apply(lambda x: get_avg_vector(x, model)).values)

joblib.dump(vectors, "quora_doc_vectors.joblib")

print("تم حفظ مصفوفة تمثيلات الوثائق في: quora_doc_vectors.joblib")
