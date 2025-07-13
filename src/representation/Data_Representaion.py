from sklearn.feature_extraction.text import TfidfVectorizer

#####################################################
create tfidf vectorizer & matrix
def create_tfidf_representation(data): 
    vectorizer = TfidfVectorizer(max_df=0.155,ngram_range=(1, 4), stop_words='english')
    data['text'] = data['text'].fillna("")  
    tfidf_matrix = vectorizer.fit_transform(data['text'])
    return tfidf_matrix, vectorizer
#####################################################


#####################################################
# create tfidf vectorizer & matrix for query refinment
def create_tfidf_representation(data): #>>>> to build refinment vectorizer 
    vectorizer = TfidfVectorizer(ngram_range=(1,4))
    vectorizer.fit_transform(data['text'])
    return vectorizer
#####################################################


