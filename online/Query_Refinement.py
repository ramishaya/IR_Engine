import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from spellchecker import SpellChecker
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from joblib import load
import spacy


# إذا كان ملف joblib يحتوي على tuple (vectorizer, matrix):
   # إذا كان هو وحده

#####################################################################################################################################
# corect the query 
nlp = spacy.load('en_core_web_sm')
def correct_spelling(query):
    spell = SpellChecker()
    corrected_query = " ".join([spell.correction(word) if spell.correction(word) is not None else word for word in query.split()])
    return corrected_query
# ================== 🔧 أدوات المعالجة ==================


# ================== ✅ دالة الاقتراح ==================

def suggest_queries(query, vectorizer, top_n=10):
    doc = nlp(query)
    suggestions = []
    for token in doc:
        if token.text in vectorizer.vocabulary_:
            index = vectorizer.vocabulary_[token.text]
            similar_terms = vectorizer.get_feature_names_out()[index:index+top_n]
            suggestions.extend(similar_terms)
    return suggestions # إزالة التكرار


# ================== 🧪 تجربة محلية ==================



