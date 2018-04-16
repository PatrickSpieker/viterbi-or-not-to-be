# Parses bc3 corpus

import numpy as np
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from functools import reduce

DEBUG = True
CORPUS = 'baseline/data/corpus-tiny.xml'
ANNOTATIONS = 'baseline/data/annotations-tiny.xml'

def main():
    with open(CORPUS, 'r') as corpus_file, open(ANNOTATIONS, 'r') as annotations_file:
        annotations = parse_annotations(annotations_file)
        threads, thread_labels = parse_corpus(corpus_file, annotations)
        sentence_features = calculate_features(threads)
        model = train_model(sentence_features, thread_labels)

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
    tree = ET.parse(xml_file)
    root = tree.getroot()

    threads = []
    thread_labels = []

    for thread in root:
        thread_text = []
        sentence_labels = []
        listno = thread[1].text # TODO: use find syntax
        debug('---------- Thread with name "' + thread[0].text + '" and listno ' + thread[1].text + ' ----------')
        
        for docnum in range(2, len(thread.getchildren())):
            # { Received, From, To, (Cc), Subject, Text }
            for item in thread[docnum]:
                if item.tag == 'Subject':
                    debug('\n    Email subject: "' + thread[docnum][3].text + '"')
                if item.tag == 'Text':
                    for sent in item:
                        debug('        Sentence id: ' + sent.attrib['id'])
                        sentence_id = sent.attrib['id']
                        sentence_labels.append(1 if sentence_id in annotations[listno] else 0)
                        debug('        Sentence: "' + sent.text + '"')
                        thread_text.append(sent.text)

        debug('\n')
        threads.append(thread_text)
        thread_labels.append(sentence_labels)

    return threads, thread_labels

def calculate_features(threads):
    documents = [' '.join(x) for x in threads]

    # Compute TF-IDF
    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf = tf_idf_vectorizer.fit_transform(documents)
    tf_idf_features = np.squeeze(np.asarray(np.mean(tf_idf, axis=1)))

    # Generate sentence features
    num_of_sentences = reduce(lambda s, thread: s + len(thread), threads, 0)
    global_sentence_index = 0

    sentence_features = np.ndarray(shape=(num_of_sentences, 3))

    for thread_index, thread in enumerate(threads):
        for sentence_index, sentence in enumerate(thread):

            # TF-IDF
            sentence_features[global_sentence_index, 0] = tf_idf_features[thread_index]
            # Sentence Length
            sentence_features[global_sentence_index][1] = len(sentence)
            # Sentence Position
            sentence_features[global_sentence_index][2] = sentence_index

            global_sentence_index += 1

    debug(sentence_features)
    return sentence_features

def train_model(sentence_features, thread_labels):
    # Flatten the thread_labels to produce sentence labels
    sentence_labels = [label for thread in thread_labels for label in thread]

    debug(sentence_labels)

    # Train the Naive Bayes model
    model = GaussianNB()
    model.fit(sentence_features, sentence_labels)

    return model

if __name__ == '__main__':
    main()
