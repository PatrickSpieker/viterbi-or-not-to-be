import argparse
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import linear_kernel
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from functools import reduce
from scipy import spatial
from nltk import tokenize
from nltk import tag
from nltk.corpus import stopwords
import configuration as config
from conversation.parsers.EmailParser import EmailParser
from conversation.feature_vectorizers.EmailFeatureVectorizer import EmailFeatureVectorizer
from Evaluation import Evaluation

def main():
    parser = argparse.ArgumentParser(description='Run the conversation-specific summarization model.')
    parser.add_argument('--type', options=['email', 'chat'])
    parser.add_argument('--model', options=['naivebayes', 'decisiontree', 'perceptron'])
    parser.add_argument('--dataset')
    parser.add_argument('--metric', options=['L', '1', '2', 'all'])
    args = parser.parse_args()

    if args['type'] == 'email':
        parser = EmailParser(args['dataset'], 'train')        
        feature_vectorizer = EmailFeatureVectorizer()
    elif args['type'] == 'chat':
        pass

    evaluation = Evaluation()

    parsed_training_data, parsed_training_labels, parsed_training_names = parser.parse()
    sentence_features = feature_vectorizer.vectorize(parsed_training_data, parsed_training_labels, parsed_training_names)
    model = train_model(sentence_features, parsed_training_labels)

    test_model(model, parsed_testing_data, parsed_testing_labels)

    evaluation.rouge_evaluation()

def train_model(sentence_features, thread_labels):
    # Flatten the thread_labels to produce sentence labels
    doc_labels = flatten(thread_labels)
    sentence_labels = flatten(doc_labels)

    # Train the model
    model = GaussianNB()
    #model = tree.DecisionTreeClassifier()
    #model = MLPClassifier()
    model.fit(sentence_features, sentence_labels)

    return model
        
def test_model():
    output_dir = config.OUTPUT + config.SYSTEM
    if os.path.exists(output_dir):
        for f in glob.glob(output_dir + '*.txt'):
            os.remove(f)
    else:
        os.makedirs(output_dir)

    with open(config.DATA_DIR + config.CORPUS + config.VALIDATION, 'r') as corpus_file, open(config.DATA_DIR + config.ANNOTATIONS + config.VALIDATION, 'r') as annotations_file:
        annotations = parse_annotations(annotations_file)
        threads, thread_labels, thread_names = parse_corpus(corpus_file, annotations)
        sentence_features = calculate_features(threads, thread_names)

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

if __name__ == '__main__':
    main()