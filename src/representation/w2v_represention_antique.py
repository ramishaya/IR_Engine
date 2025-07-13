import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import joblib

# 1. تحميل بيانات الوثائق
df = pd.read_csv("cleaned_antique.csv")

# ✅ معالجة القيم الفارغة:
df['cleaned_text'] = df['cleaned_text'].fillna("")

# 2. تحميل موديل Word2Vec
model = Word2Vec.load("word2vec_antique_200.model")

# 3. دالة لتحويل النص إلى متوسط متجهات الكلمات
def get_avg_vector(text, model):
    words = text.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# 4. إنشاء تمثيلات الوثائق
vectors = np.vstack(df['cleaned_text'].apply(lambda x: get_avg_vector(x, model)).values)

# 5. حفظ التمثيلات باستخدام joblib
joblib.dump(vectors, "antique_doc_vectors_2000.joblib")

print("✅ تم حفظ مصفوفة تمثيلات الوثائق في: antique_doc_vectors_200.joblib")
