# Baseline Naive Bayes model using the BC3 corpus

import numpy as np
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import linear_kernel
from functools import reduce
from scipy import spatial

DEBUG = True
DATA_DIR = 'data/'

CORPUS = 'corpus-tiny'
ANNOTATIONS = 'annotations-tiny'

TRAIN = '.train.xml'
VALIDATION = '.val.xml'
TEST = '.test.xml'

def main():
    with open(DATA_DIR + CORPUS + TRAIN, 'r') as corpus_file, open(DATA_DIR + ANNOTATIONS + TRAIN, 'r') as annotations_file:
        annotations = parse_annotations(annotations_file)
        threads, thread_labels, thread_names = parse_corpus(corpus_file, annotations)
        sentence_features = calculate_features(threads, thread_names)
        model = train_model(sentence_features, thread_labels)
        model_score = evaluate_model(model)

def debug(output):
    if DEBUG:
        print(output)

def parse_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = {}

    for thread in root:
        listno = thread.find('listno').text

        # Note: only using the first annotation for now. There are multiple
        # annotations available for each email thread.
        annotation = thread.find('annotation')
        sentence_ids = []
        for item in annotation.find('sentences'):
            sentence_ids.append(item.attrib['id'])
        
        # Associate the listno with the list of extractive summary sentences
        annotations[listno] = sentence_ids

    return annotations

def parse_corpus(xml_file, annotations):
    # Parse xml data into tree
    tree = ET.parse(xml_file)
    root = tree.getroot()

    threads = []
    thread_labels = []
    thread_names = []

    for thread in root:
        thread_text = []
        sentence_labels = []
        name = thread.find('name').text
        listno = thread.find('listno').text
        debug('---------- Thread with name "' + name + '" and listno ' + listno + ' ----------')
        
        for doc in thread.findall('DOC'):
            # Email doc contents typically contain { Received, From, To, (Cc), Subject, Text }
            subject = doc.find('Subject').text
            text = doc.find('Text')
            debug('\n    Email subject: "' + subject + '"')
            for sent in text:
                debug('        Sentence id: ' + sent.attrib['id'])
                sentence_id = sent.attrib['id']
                sentence_labels.append(1 if sentence_id in annotations[listno] else 0)
                debug('        Sentence: "' + sent.text + '"')
                thread_text.append(sent.text)

        debug('\n')
        threads.append(thread_text)
        thread_labels.append(sentence_labels)
        thread_names.append(name)

    return threads, thread_labels, thread_names

def calculate_features(threads, thread_names):
    documents = [' '.join(x) for x in threads]

    # Compute TF-IDF
    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf = tf_idf_vectorizer.fit_transform(documents)
    tf_idf_features = np.squeeze(np.asarray(np.mean(tf_idf, axis=1)), axis=1)

    # Generate sentence features
    num_of_sentences = reduce(lambda s, thread: s + len(thread), threads, 0)
    global_sentence_index = 0

    sentence_features = np.ndarray(shape=(num_of_sentences, 5))

    for thread_index, thread in enumerate(threads):

        # Compute TF-ISF for thread name (index 0) and thread content
        thread.append(thread_names[thread_index])
        tf_isf_vectorizer = TfidfVectorizer()
        tf_isf = tf_isf_vectorizer.fit_transform(thread)
        tf_isf_features = np.squeeze(np.asarray(np.mean(tf_isf, axis=1)))
        title_vector = tf_isf[len(thread) - 1]

        for sentence_index, sentence in enumerate(thread[:len(thread)-1]):

            # TF-IDF
            sentence_features[global_sentence_index, 0] = tf_idf_features[thread_index]
            # TF-ISF
            sentence_features[global_sentence_index, 1] = tf_isf_features[sentence_index]
            # Sentence Length
            sentence_features[global_sentence_index, 2] = len(sentence)
            # Sentence Position
            sentence_features[global_sentence_index, 3] = sentence_index
            # Similarity to Title
            sentence_vector = tf_isf[sentence_index]
            sentence_features[global_sentence_index, 4] = linear_kernel(title_vector, sentence_vector).flatten()

            global_sentence_index += 1

    debug(sentence_features)
    return sentence_features

def train_model(sentence_features, thread_labels):
    # Flatten the thread_labels to produce sentence labels
    sentence_labels = flatten(thread_labels)

    debug(sentence_labels)

    # Train the Naive Bayes model
    model = GaussianNB()
    model.fit(sentence_features, sentence_labels)

    return model

def evaluate_model(model):
    with open(DATA_DIR + CORPUS + VALIDATION, 'r') as corpus_file, open(DATA_DIR + ANNOTATIONS + VALIDATION, 'r') as annotations_file:
        annotations = parse_annotations(annotations_file)
        threads, thread_labels, thread_names = parse_corpus(corpus_file, annotations)
        sentence_features = calculate_features(threads, thread_names)

        predicted_annotations = model.predict(sentence_features)
        # Flatten the thread_labels to produce sentence labels
        actual_annotations = flatten(thread_labels)

        sentences = flatten(threads)

        print('Sentences: ')
        for sentence in sentences:
            print(sentence)
        print('Predicted summary: ')
        for index, sentence in enumerate(sentences):
            if (predicted_annotations[index] == 1):
                print(sentence)
        print('Actual summary: ')
        for index, sentence in enumerate(sentences):
            if (actual_annotations[index] == 1):
                print(sentence)

    return 0 # TODO: use proper accuracy metric

def flatten(nested_list):
    return [label for thread in nested_list for label in thread]

if __name__ == '__main__':
    main()
