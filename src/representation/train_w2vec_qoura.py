import pandas as pd
from gensim.models import Word2Vec

df = pd.read_csv('cleaned_quora.csv')
df['cleaned_text'] = df['cleaned_text'].fillna('')

sentences = [text.split() for text in df['cleaned_text']]

# train Word2Vec model
model = Word2Vec(
    sentences=sentences,
    vector_size=200,   # حجم التمثيل الشعاعي
    window=5,          # حجم النافذة السياقية
    min_count=2,       # تجاهل الكلمات التي تكررت أقل من مرتين
    workers=4,         # عدد الأنوية المستخدمة
    sg=1               # 1 لـ Skip-gram، 0 لـ CBOW
)

model.save("word2vec_quora.model")

print(" تم تدريب النموذج وحفظه في word2vec_quora.model")
