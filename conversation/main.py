import argparse
import glob
import os
import pdb
from functools import reduce

from nltk import tag, tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import configuration as config
from feature_vectorizers.EmailFeatureVectorizer import \
    EmailFeatureVectorizer
from preprocessors.EmailPreprocessor import EmailPreprocessor
from parsers.EmailParser import EmailParser
from evaluation.Evaluation import Evaluation
from scipy import spatial

def main():
    parser = argparse.ArgumentParser(description='Run the conversation-specific summarization model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', help='the path of the dataset to use, specified as a path relative to the data/ directory, i.e. to use the full bc3 dataset: \'bc3/full\'')
    parser.add_argument('--type', choices=['email', 'chat'], default='email', help='the format of the dataset being used')
    parser.add_argument('--model', choices=['naivebayes', 'decisiontree', 'perceptron'], default='naivebayes', help='the type of model to train and test')
    parser.add_argument('--metric', choices=['L', '1', '2', 'all'], default='all', help='which metric(s) to report when evaluating the model')
    parser.add_argument('--debug', action='store_true', help='if set, outputs various debugging information during execution')
    parser.add_argument('--examples', action='store_true', help='if set, displays the system-generated summaries during evaluation')
    args = parser.parse_args()

    # Make the dataset relative to the data folder
    dataset = '../data/' + args.dataset

    # Use the appropriate parser, preprocessor, and feature vectorizer for the desired data type
    if args.type == 'email':
        parser = EmailParser(dataset)        
        preprocessor = EmailPreprocessor()
        feature_vectorizer = EmailFeatureVectorizer()
    elif args.type == 'chat':
        pass

    # Use the appropriate model type
    if args.model == 'naivebayes':
        model_type = GaussianNB
    elif args.model == 'decisiontree':
        model_type = DecisionTreeClassifier
    elif args.model == 'perceptron':
        model_type = MLPClassifier

    # Use the appropriate evaluation metrics
    if args.metric == 'all':
        metrics = ['L', '1', '2']
    else:
        metrics = [args.metric]
    evaluation = Evaluation(metrics, debug=args.debug, examples=args.examples)

    # Parse training data
    training_data = parser.parse('train')

    # Preprocess training data
    training_data = preprocessor.preprocess(training_data)

    # Produce sentence features
    training_sentence_features = feature_vectorizer.vectorize(training_data)

    # Train model using training data
    model = train_model(model_type, training_data, training_sentence_features)

    # Parse val data
    val_data = parser.parse('val')

    # Preprocess val data
    val_data = preprocessor.preprocess(val_data)

    # Produce sentence features
    val_sentence_features = feature_vectorizer.vectorize(val_data)

    # Generate the model's predicted summaries on the val data
    test_model(model, val_data, val_sentence_features)

    # Compile the reference summaries
    parser.compile_reference_summaries()

    # Evaluate the model's performance using the preferred metrics
    evaluation.rouge_evaluation()

def train_model(model_type, training_data, sentence_features):
    thread_labels = training_data['labels']

    # Flatten the thread_labels to produce sentence labels
    sentence_labels = flatten(thread_labels)

    # Train the model
    model = model_type()
    model.fit(sentence_features, sentence_labels)

    return model
        
def test_model(model, val_data, sentence_features):
    output_dir = config.OUTPUT + config.SYSTEM
    if os.path.exists(output_dir):
        for f in glob.glob(output_dir + '*.txt'):
            os.remove(f)
    else:
        os.makedirs(output_dir)

    threads = val_data['data']

    predicted_annotations = model.predict(sentence_features)
    sentences = flatten(threads)

    sentence = 0
    for thread_index, thread in enumerate(threads):
        thread_summary = []
        for _ in range(0, len(thread), 3):
            if predicted_annotations[sentence] == 1:
                thread_summary.append(sentences[sentence] + ' ')
            sentence += 3
        
        filename = output_dir + 'thread{}_system1.txt'.format(thread_index)
        with open(filename, 'w+') as output_file:
            output_file.write(''.join(thread_summary))

def flatten(nested_list):
    return [label for thread in nested_list for label in thread]

if __name__ == '__main__':
    main()
