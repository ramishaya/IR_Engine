from scipy import sparse
from Data_Representaion import create_tfidf_representation
import joblib
#####################################################################
save vectorizer & matrix
def save_index(tfidf_matrix, vectorizer, index_file, vectorizer_file):
    sparse.save_npz(index_file, tfidf_matrix)
    with open(vectorizer_file, 'wb') as f:
        joblib.dump(vectorizer, f)
# #####################################################################

# #####################################################################
# load vectorizer & matrix
def load_index(index_file, vectorizer_file):
    tfidf_matrix = sparse.load_npz(index_file)
    with open(vectorizer_file, 'rb') as f:
        vectorizer = joblib.load(f)
    return tfidf_matrix, vectorizer
#####################################################################


############################################### WIKI ###############################################

import pandas as pd
data_quora = pd.read_csv('quora.csv')
data_quora.fillna('defualt value')
tfidf_matrix_wiki , vectorizer_wiki = create_tfidf_representation(data_quora)
save_index(tfidf_matrix_wiki, vectorizer_wiki, 'tfidf_quora.npz', 'victorizer_quora.joblib')


# import pandas as pd # >>>> for query refinment 
# data_wiki = pd.read_csv('wiki_queries.csv')
# data_wiki.fillna('defualt value')
# vectorizer_wiki = create_tfidf_representation(data_wiki)
# with open('wiki_for_query_refinment.pkl', 'wb') as f:
#         pickle.dump(vectorizer_wiki, f)

############################################### antique ###############################################

# import pandas as pd
# data_antique = pd.read_csv('antique.csv')
# data_antique.fillna('defualt value')
# tfidf_matrix_antique , vectorizer_antique = create_tfidf_representation(data_antique)
# save_index(tfidf_matrix_antique, vectorizer_antique, 'tfidf_antique.npz', 'victorizer_antique.joblib')


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# تحميل بيانات Quora
data_antique = pd.read_csv('antique_queries.csv')

# التأكد من عدم وجود قيم مفقودة في العمود المناسب (مثلاً: 'question' أو 'text')
data_antique.fillna('default value', inplace=True)

# اختيار العمود الذي يحتوي الاستعلامات
corpus = data_antique['text'].astype(str).tolist()  # ✅ غيّر 'question' إذا كان اسم العمود مختلفًا

# إنشاء وتدريب الـ Vectorizer
vectorizer_antique = TfidfVectorizer()
vectorizer_antique.fit(corpus)

# حفظ النموذج
joblib.dump(vectorizer_antique, 'antique_for_query_refinment.joblib')

print("✅ تم إنشاء الملف antique_for_query_refinment بنجاح.")


