# Baseline Naive Bayes model using the BC3 corpus

import configuration as config
import numpy as np
import xml.etree.ElementTree as ET
import os
import glob
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import linear_kernel
from functools import reduce
from scipy import spatial
from nltk import tokenize
from nltk import tag

def main():
    with open(config.DATA_DIR + config.CORPUS + config.TRAIN, 'r') as corpus_file, open(config.DATA_DIR + config.ANNOTATIONS + config.TRAIN, 'r') as annotations_file:
        annotations = parse_annotations(annotations_file)
        threads, thread_labels, thread_names = parse_corpus(corpus_file, annotations)
        sentence_features = calculate_features(threads, thread_names)
        model = train_model(sentence_features, thread_labels)
        evaluate_model(model)

def debug(output):
    if config.DEBUG:
        print(output)

def parse_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotation_map = {}

    for thread in root:
        listno = thread.find('listno').text
        annotation_map[listno] = []

        annotations = thread.findall('annotation')

        for annotation in annotations:
            sentence_ids = []
            for item in annotation.find('sentences'):
                sentence_ids.append(item.attrib['id'])
            # Associate the listno with the list of extractive summary sentences
            annotation_map[listno].append(sentence_ids)

    return annotation_map

def parse_corpus(xml_file, annotations):
    # Parse xml data into tree
    tree = ET.parse(xml_file)
    root = tree.getroot()

    threads = []
    thread_labels = []
    thread_names = []

    for thread in root:
        thread_text = []
        doc_labels = []
        name = thread.find('name').text
        listno = thread.find('listno').text
        debug('---------- Thread with name "' + name + '" and listno ' + listno + ' ----------')
        
        for doc in thread.findall('DOC'):
            # Email doc contents typically contain { Received, From, To, (Cc), Subject, Text }
            subject = doc.find('Subject').text
            text = doc.find('Text')
            debug('\n    Email subject: "' + subject + '"')
            sentence_labels = []

            for sent in text:
                for annotation in annotations[listno]:
                    debug('        Sentence id: ' + sent.attrib['id'])
                    sentence_id = sent.attrib['id']
                    sentence_labels.append(1 if sentence_id in annotation else 0)
                    debug('        Sentence: "' + sent.text + '"')
                    thread_text.append(sent.text)
            doc_labels.append(sentence_labels)

        debug('\n')
        threads.append(thread_text)
        thread_labels.append(doc_labels)
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

    sentence_features = np.ndarray(shape=(num_of_sentences, 9))

    for thread_index, thread in enumerate(threads):

        # Compute TF-ISF for thread name (index 0) and thread content
        thread_with_name = thread.copy()
        thread_with_name.append(thread_names[thread_index])
        tf_isf_vectorizer = TfidfVectorizer()
        tf_isf = tf_isf_vectorizer.fit_transform(thread_with_name)
        tf_isf_features = np.squeeze(np.asarray(np.mean(tf_isf, axis=1)))
        title_vector = tf_isf[len(thread_with_name) - 1]

        # Count number of special terms in thread
        special_counts = []
        total_special_count = 0.0
        for sentence in thread:
            sent_tokens = tokenize.word_tokenize(sentence)
            tagged_sent = tag.pos_tag(sent_tokens)

            prev_proper_index = 1
            sent_special_count = 0.0

            for word_index, tagged_word in enumerate(tagged_sent):
                pos = tagged_word[1]
                if pos == 'NNP':
                    if prev_proper_index != word_index - 1:
                        sent_special_count = sent_special_count + 1.0
                    prev_proper_index = word_index
                elif pos == 'CD':
                    sent_special_count = sent_special_count + 1.0
            
            special_counts.append(sent_special_count)
            total_special_count = total_special_count + sent_special_count

        for sentence_index, sentence in enumerate(thread):
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
            # Centroid Coherence
            tf_isf_mean = np.mean(tf_isf, axis=0)
            sentence_features[global_sentence_index, 5] = linear_kernel(tf_isf_mean, sentence_vector).flatten()
            # Special Terms
            sentence_features[global_sentence_index, 6] = special_counts[sentence_index] / total_special_count
            # Special Case: Starts with '>'
            sentence_features[global_sentence_index, 7] = 1 #0 if sentence.startswith('>') else 1
            # Position from the end of the email
            sentence_features[global_sentence_index, 8] = 1 #len(thread) - sentence_index

            global_sentence_index += 1

    debug(sentence_features)
    return sentence_features

def train_model(sentence_features, thread_labels):
    # Flatten the thread_labels to produce sentence labels
    doc_labels = flatten(thread_labels)
    sentence_labels = flatten(doc_labels)

    debug(sentence_features.shape)
    debug(len(sentence_labels))

    # Train the Naive Bayes model
    model = GaussianNB()
    model.fit(sentence_features, sentence_labels)

    return model

def evaluate_model(model):

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
            for _ in range(len(thread)):
                if predicted_annotations[sentence] == 1:
                    thread_summary.append(sentences[sentence])
                sentence += 1
            
            filename = output_dir + 'thread{}_system1.txt'.format(thread_index)
            with open(filename, 'w+') as output_file:
                output_file.write(''.join(thread_summary))

def flatten(nested_list):
    return [label for thread in nested_list for label in thread]

if __name__ == '__main__':
    main()
