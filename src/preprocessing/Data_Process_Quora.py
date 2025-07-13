import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from src.preprocessing.Additional_StopWords_Quora import additional_stopwords
from src.preprocessing.Abbreviations_Quora import abbreviations
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    return [word.lower() for word in words]

def remove_punctuation(text):
    """Remove punctuation from text and replace with spaces"""
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return text.translate(translator)

def rephrasing_abbreviations(text):
    """Rephrase abbreviations in the text"""
    for abbreviation, basic_sentence in abbreviations.items():
        text = re.sub(r'\b{}\b'.format(abbreviation), basic_sentence, text)
    return text

def data_cleaning(text):
    """Clean data by removing URLs, repeated characters, and unwanted symbols"""
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
    text = re.sub(r'\b(\w+)(?:\s+\1)+\b', r'\1', text)
    text = re.sub(r'\$', " dollar ", text)
    text = re.sub(r'\#', " hash ", text)
    text = re.sub(r'\*', " star ", text)
    text = re.sub(r'\%', " percent ", text)
    text = re.sub(r'\&', " and ", text)
    text = re.sub(r'\|', " or ", text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\b\w{21,}\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def tokenize_text(text):
    """Tokenize text into words"""
    return word_tokenize(text)

def remove_stopwords(tokens):
    """Remove stopwords from list of tokens"""
    return [word for word in tokens if word.lower() not in stopwords.words('english') and word.lower() not in additional_stopwords]

def stemming(tokens):
    """Stem words to their root form"""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def get_wordnet_pos(tag):
    """Map POS tag to first character lemmatize() accepts"""
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag[0].upper(), wordnet.NOUN)

def data_lemmatization(tokens):
    """Lemmatize words to their root form"""
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokens)
    return [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]
def data_processing_quora(text):
    """Process text data through several cleaning and tokenizing steps"""
    if not isinstance(text, str):  # ✅ لمنع الخطأ إن كانت القيمة NaN أو float
        return ""

    text = rephrasing_abbreviations(text)
    text = remove_punctuation(text)
    text = data_cleaning(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    tokens = to_lowercase(tokens)
    tokens = data_lemmatization(tokens)
    return " ".join(tokens)


# Test the data_processing function
# print(data_processing('Hi I am Karam , now we are Testing the Data processssingg function , here we havr the misstak in the textt , do you have any problems ? IN USA UK'))
