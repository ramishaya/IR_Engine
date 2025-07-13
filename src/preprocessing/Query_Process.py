from src.preprocessing.Data_Process_Quora import data_processing_quora
from src.preprocessing.Data_Process_Antique import data_processing_antique

#######################################################################################
# process the query using the dataset processer
def process_query(data, query, vectorizer):
    if data == 'quora':
        processed_query = data_processing_quora(query)
    else:
        processed_query = data_processing_antique(query)
    query_vector = vectorizer.transform([processed_query])
    return query_vector
#######################################################################################
