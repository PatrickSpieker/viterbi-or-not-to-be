from flask import Flask
from flask import request
from flask_cors import CORS

from preprocessors.EmailPreprocessor import EmailPreprocessor
from preprocessors.ChatPreprocessor import ChatPreprocessor
from feature_vectorizers.EmailFeatureVectorizer import \
    EmailFeatureVectorizer
from feature_vectorizers.ChatFeatureVectorizer import \
    ChatFeatureVectorizer

from abstractive import generate_formatted, flatten

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
    return 'Ya this API is working! Now go use our demo!'

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

    all_sentences = flatten(flatten(preprocessed['data']))
    all_authors = flatten(flatten(preprocessed['authors']))
    formatted = generate_formatted(all_sentences, all_authors)

    feature_values = {}
    for feature_index, feature in enumerate(ChatFeatureVectorizer.FEATURES):
        feature_values[feature] = list(features[:,feature_index])

    response_data = {
        'predictions': predictions_list,
        'formatted': formatted,
        'features': feature_values
    }

    return json.dumps(response_data)