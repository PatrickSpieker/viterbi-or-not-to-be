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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
import numpy as np

from feature_vectorizers.EmailFeatureVectorizer import \
    EmailFeatureVectorizer
from feature_vectorizers.ChatFeatureVectorizer import \
    ChatFeatureVectorizer
from preprocessors.EmailPreprocessor import EmailPreprocessor
from preprocessors.ChatPreprocessor import ChatPreprocessor
from parsers.EmailParser import EmailParser
from parsers.ChatParser import ChatParser
from evaluation.Evaluation import Evaluation
from scipy import spatial

# The directory where results should be output
OUTPUT = 'output/'

# The subdirectories under which ROUGE-compatible summaries should be output
REFERENCE = 'reference/'
SYSTEM = 'system/'

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate the conversation-specific model for automatic conversation summarization. Allows selection between several different types of models and datasets, as well as customization of the metrics to be used in evaluation and various options for debugging. After evaluation, system-generated summaries can be found in the output/system directory', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', help='the path of the dataset to use, specified as a path relative to the data/ directory, i.e. to use the full bc3 dataset: \'bc3/full\'')
    parser.add_argument('--type', choices=['email', 'chat'], default='email', help='the format of the dataset being used')
    parser.add_argument('--model', choices=['naivebayes', 'decisiontree', 'perceptron', 'regression_dt', 'regression_br'], default='naivebayes', help='the type of model to train and test')
    parser.add_argument('--metric', choices=['L', '1', '2', 'all'], default='all', help='which metric(s) to report when evaluating the model')
    parser.add_argument('--threshold', type=float, default=0.5, help='the cutoff at which to consider sentences as part of the summary')
    parser.add_argument('--evaldataset', help='the path of the dataset to use for evaluation')
    parser.add_argument('--evaltype', choices=['email', 'chat'], help='the format of the evaluation dataset being used')
    parser.add_argument('--debug', action='store_true', help='if set, outputs various debugging information during execution')
    parser.add_argument('--examples', action='store_true', help='if set, displays the system-generated summaries during evaluation')
    parser.add_argument('--nopreprocessing', action='store_true', help='if set, disables preprocessing for the training and validation data')
    parser.add_argument('--crosstrain', help='if specified, the dataset to additionally train on before evaluation (assumed to be the opposite type of data from the argument given in "type")')
    args = parser.parse_args()

    # Interpret the dataset relative to the data folder
    dataset = '../data/' + args.dataset

    # Use the appropriate parser, preprocessor, and feature vectorizer for the desired data type
    if args.type == 'email':
        parser = EmailParser(dataset, args.debug)
        preprocessor = EmailPreprocessor()
        feature_vectorizer = EmailFeatureVectorizer()
    elif args.type == 'chat':
        parser = ChatParser(dataset, args.debug)
        preprocessor = ChatPreprocessor()
        feature_vectorizer = ChatFeatureVectorizer()

    # If specified, use alternate evaluation
    if args.evaldataset is not None and args.evaltype is not None:
        eval_type = args.evaltype
        eval_dataset = '../data/' + args.evaldataset

        if args.evaltype == 'email':
            eval_parser = EmailParser(eval_dataset, args.debug)
            eval_preprocessor = EmailPreprocessor()
            eval_feature_vectorizer = EmailFeatureVectorizer()
        elif args.evaltype == 'chat':
            eval_parser = ChatParser(eval_dataset, args.debug)
            eval_preprocessor = ChatPreprocessor()
            eval_feature_vectorizer = ChatFeatureVectorizer()
    else:
        eval_type = args.type
        eval_parser = parser
        eval_preprocessor = preprocessor
        eval_feature_vectorizer = feature_vectorizer

    # If cross-training is enabled, create the second set of training input processors
    if args.crosstrain is not None:
        if args.type == 'chat':
            cross_parser = EmailParser(dataset, args.debug)
            cross_preprocessor = EmailPreprocessor()
            cross_feature_vectorizer = EmailFeatureVectorizer()
        elif args.type == 'email':
            cross_parser = ChatParser(args.crosstrain, args.debug)
            cross_preprocessor = ChatPreprocessor()
            cross_feature_vectorizer = ChatFeatureVectorizer()

    # Use the appropriate model type
    if args.model == 'naivebayes':
        model = GaussianNB()
    elif args.model == 'decisiontree':
        model = DecisionTreeClassifier()
    elif args.model == 'perceptron':
        model = MLPClassifier()
    elif args.model == 'regression_dt':
        model = DecisionTreeRegressor()
    elif args.model == 'regression_br':
        model = BayesianRidge(compute_score=True)

    # Use the appropriate evaluation metrics
    if args.metric == 'all':
        metrics = ['L', '1', '2']
    else:
        metrics = [args.metric]
    evaluation = Evaluation(metrics, debug=args.debug, examples=args.examples)

    # Use the appropriate threshold
    threshold = args.threshold

    # Parse training data
    training_data = parser.parse('train')

    # Preprocess training data
    if (not args.nopreprocessing):
        training_data = preprocessor.preprocess(training_data)

    # Produce sentence features
    training_sentence_features = feature_vectorizer.vectorize(training_data)

    # Flatten thread labels to match the feature shape
    thread_labels = flatten(flatten(training_data['labels']))

    # If crosstraining is enabled, use the second round of training data
    if args.crosstrain is not None:
        cross_training_data = cross_parser.parse('train')
        if (not args.nopreprocessing):
            cross_training_data = cross_preprocessor.preprocess(cross_training_data)
        cross_training_sentence_features = cross_feature_vectorizer.vectorize(cross_training_data)

        # Concatenate so as to train on all available data
        training_sentence_features = np.concatenate(training_sentence_features, cross_training_sentence_features)
        thread_labels = thread_labels.extend(flatten(flatten(cross_training_data['labels'])))

    # Train model using training data
    model.fit(training_sentence_features, thread_labels)

    # Parse val data
    val_data = eval_parser.parse('val')

    # Preprocess val data
    if (not args.nopreprocessing):
        val_data = eval_preprocessor.preprocess(val_data)

    # Produce sentence features
    val_sentence_features = eval_feature_vectorizer.vectorize(val_data)

    # Generate the model's predicted summaries on the val data
    test_model(model, val_data, val_sentence_features, 3 if eval_type == 'email' else 1, threshold)

    # Compile the reference summaries
    eval_parser.compile_reference_summaries()

    # Evaluate the model's performance using the preferred metrics
    evaluation.rouge_evaluation()
        
def test_model(model, val_data, sentence_features, step, threshold):
    output_dir = OUTPUT + SYSTEM
    if os.path.exists(output_dir):
        for f in glob.glob(output_dir + '*.txt'):
            os.remove(f)
    else:
        os.makedirs(output_dir)

    threads = val_data['data']
    collapsed_threads = collapse_threads(val_data['data'])

    predicted_annotations = model.predict(sentence_features)
    sentences = flatten(collapsed_threads)

    sentence = 0
    for thread_index, thread in enumerate(collapsed_threads):
        thread_summary = []
        for _ in range(0, len(thread), step):
            if predicted_annotations[sentence] > threshold:
                thread_summary.append(sentences[sentence] + ' ')
            sentence += step
        
        filename = output_dir + 'thread{}_system1.txt'.format(thread_index)
        with open(filename, 'w+') as output_file:
            output_file.write('\n'.join(thread_summary))

def flatten(nested_list):
    return [label for thread in nested_list for label in thread]

def collapse_threads(threads):
    collapsed_threads = []
    for thread in threads:
        collapsed_threads.append(flatten(thread))
    return collapsed_threads

if __name__ == '__main__':
    main()
