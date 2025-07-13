from flask import Flask, request, jsonify
import pandas as pd
from src.preprocessing.Data_Process_Antique import data_processing_antique
from src.preprocessing.Data_Process_Quora import data_processing_quora

app = Flask(__name__)
@app.route('/clean_text', methods=['POST'])
def clean_text():
    data = request.get_json()
    dataset = data.get('dataset')
    text = data.get('text')

    if dataset == 'antique':
        cleaned = data_processing_antique(text)
    elif dataset == 'quora':
        cleaned = data_processing_quora(text)
    else:
        return jsonify({'error': 'Invalid dataset'}), 400

    return jsonify({'cleaned_text': cleaned})


@app.route('/get_cleaned_data', methods=['GET'])
def get_cleaned_data():
    dataset = request.args.get('dataset')  # ?dataset=antique
    if dataset == 'antique':
        df = pd.read_csv("cleaned_antique.csv")
    elif dataset == 'quora':
        df = pd.read_csv("cleaned_quora.csv")
    else:
        return jsonify({'error': 'Invalid dataset'}), 400

    # حوّل الـ DataFrame إلى list of dicts
    records = df.to_dict(orient='records')
    return jsonify({'data': records})


if __name__ == '__main__':
    app.run(port=5001, debug=True)















    