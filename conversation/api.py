from flask import Flask
from flask import request
from flask_cors import CORS

from preprocessors.EmailPreprocessor import EmailPreprocessor
from preprocessors.ChatPreprocessor import ChatPreprocessor
from feature_vectorizers.EmailFeatureVectorizer import \
    EmailFeatureVectorizer
from feature_vectorizers.ChatFeatureVectorizer import \
    ChatFeatureVectorizer

import pickle
import ast
import json

app = Flask(__name__)
CORS(app)

MODEL = 'saved_models/snack_pack.pickle'
with open(MODEL, 'rb') as model_file:
    model = pickle.load(model_file)
preprocessor = ChatPreprocessor()
feature_vectorizer = ChatFeatureVectorizer()

@app.route('/')
def serve():
    return 'Ya this Flask server is running!!'

@app.route('/api', methods=['POST'])
def api():
    to_summarize = request.get_json()
    messages = to_summarize['messages']
    authors = to_summarize['authors']

    for i in range(len(messages)):
        messages[i] = '<{}> {}'.format(authors[i], messages[i])

    labels = list(map(lambda x : 0, messages))

    data = {
        'data': [[messages]],
        'authors': [],
        'labels': [[labels]],
        'names': []
    }

    print('data')
    print(data)
    print()

    preprocessed = preprocessor.preprocess(data)

    print('preprocessed')
    print(preprocessed)
    print()

    features = feature_vectorizer.vectorize(preprocessed)
    predictions = model.predict(features)

    print('predictions')
    print(predictions)
    print()

    predictions_list = list(predictions.tolist())

    response_data = {
        'predictions': predictions_list
    }

    return json.dumps(response_data)