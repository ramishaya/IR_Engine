import numpy as np
from scipy import sparse
#################################################################
# get the similar docs
def rank_documents(query_vector, tfidf_matrix):
    # calculate cosin sim
    cosine_similarities = tfidf_matrix @ query_vector.T
    # convert to array
    cosine_similarities = cosine_similarities.toarray().flatten()
    # rand docs
    ranked_indices = np.argsort(-cosine_similarities)
    return ranked_indices, cosine_similarities
#################################################################