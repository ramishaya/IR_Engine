import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from Data_Processing_Antique import data_processing_antique
import time

# 📌 تحميل بيانات الوثائق الأصلية
docs_df = pd.read_csv("antique_docs_train.csv")

# 📌 التأكد من وجود الأعمدة المطلوبة
assert 'doc_id' in docs_df.columns and 'text' in docs_df.columns, "تأكد من وجود الأعمدة doc_id و text"

# 📌 تعويض القيم الفارغة في النصوص
docs_df['text'] = docs_df['text'].fillna('')

# 📌 تنظيف النصوص
docs_df["clean_text"] = docs_df["text"].apply(data_processing_antique)

# 📌 حساب وقت التنفيذ
start_time = time.time()

# 📌 بناء تمثيل TF-IDF بنفس إعدادات التقييم
vectorizer = TfidfVectorizer(
    max_df=0.155,
    ngram_range=(1, 4),
    stop_words='english'
)
tfidf_matrix = vectorizer.fit_transform(docs_df["clean_text"])

elapsed_time = time.time() - start_time
print(f"✅ تم بناء فهرس TF-IDF في {elapsed_time:.2f} ثانية")

# 📌 حفظ تمثيل الوثائق (مصفوفة)
save_npz("antique_tfidf_vectors_final.npz", tfidf_matrix)

# 📌 حفظ الـ vectorizer
joblib.dump(vectorizer, "antique_tfidf_vectorizer_final.joblib")

print("✅ تم حفظ تمثيل الوثائق والفكتورايزر بنجاح 🎉")
