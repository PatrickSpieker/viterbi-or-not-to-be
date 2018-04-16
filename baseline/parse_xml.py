# Parses bc3 corpus

import numpy as np
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import reduce

DEBUG = False

def main():
    with open('baseline/data/corpus-tiny.xml', 'r') as xml_file:
        threads = parse_corpus(xml_file)
        sentence_features = calculate_features(threads)

def debug(output):
    if DEBUG:
        print(output)

def calculate_features(threads):
    documents = [' '.join(x) for x in threads]
    vectorizer = TfidfVectorizer()
    tf_idf = vectorizer.fit_transform(documents)

    # Should result in a vector of shape (threads)
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

def parse_corpus(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    threads = []

    for thread in root:
        thread_text = []
        debug('---------- Thread with name "' + thread[0].text + '" and listno ' + thread[1].text + ' ----------')
        
        for docnum in range(2, len(thread.getchildren())):
            # { Received, From, To, (Cc), Subject, Text }
            for item in thread[docnum]:
                if item.tag == 'Subject':
                    debug('\n    Email subject: "' + thread[docnum][3].text + '"')
                if item.tag == 'Text':
                    for sent in item:
                        debug('        Sentence id: ' + sent.attrib['id'])
                        debug('        Sentence: "' + sent.text + '"')
                        thread_text.append(sent.text)

        threads.append(thread_text)
        debug('\n')

    return threads

if __name__ == '__main__':
    main()
