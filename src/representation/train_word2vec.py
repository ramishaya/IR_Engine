import pandas as pd
from gensim.models import Word2Vec

# ========================
# 1. تحميل البيانات المعالجة
# ========================
df = pd.read_csv('cleaned_data_for_wiki_final.csv')

# تأكد من عدم وجود فراغات أو نصوص ناقصة
df['text'] = df['text'].fillna('')

# ========================
# 2. تحويل النصوص إلى قوائم كلمات (Tokenization)
# ========================
sentences = [text.split() for text in df['text']]

# ========================
# 3. تدريب نموذج Word2Vec
# ========================
model = Word2Vec(
    sentences=sentences,   # البيانات
    vector_size=100,       # حجم الشعاع (vector) لكل كلمة
    window=5,              # حجم النافذة (عدد الكلمات المحيطة)
    min_count=2,           # تجاهل الكلمات التي تظهر أقل من مرتين
    workers=4,             # عدد الأنوية المستخدمة في المعالجة (حسب جهازك)
    sg=1                   # 1 = استخدام Skip-gram, و 0 = CBOW
)

# ========================
# 4. حفظ النموذج المدرب على شكل ملف
# ========================
model.save("word2vec.model")

print("✅ تم تدريب النموذج وحفظه في word2vec.model")
