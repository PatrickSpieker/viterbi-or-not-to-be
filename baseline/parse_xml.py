# Parses bc3 corpus

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy
import functools
import xml.etree.ElementTree as ET

def calculate_features(threads):
    documents = [' '.join(x) for x in threads]
    vectorizer = TfidfVectorizer()
    tf_idf = vectorizer.fit_transform(documents)

    # Should result in a vector of shape (threads)
    tf_idf_features = numpy.squeeze(numpy.asarray(numpy.mean(tf_idf, axis=1)))

    # Generate sentence features
    num_of_sentences = functools.reduce(lambda s, thread: s + len(thread), threads, 0)
    global_sentence_index = 0

    sentence_features = numpy.ndarray(shape=(num_of_sentences, 3))

    for thread_index, thread in enumerate(threads):
        for sentence_index, sentence in enumerate(thread):

            # TF-IDF
            sentence_features[global_sentence_index, 0] = tf_idf_features[thread_index]
            # Sentence Length
            sentence_features[global_sentence_index][1] = len(sentence)
            # Sentence Position
            sentence_features[global_sentence_index][2] = sentence_index

            global_sentence_index += 1

    return sentence_features

def parseXML(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    threads = []

    for thread in root:
        thread_text = []
        print('---------- Thread with name "' + thread[0].text + '" and listno ' + thread[1].text + ' ----------')
        
        for docnum in range(2, len(thread.getchildren())):
            # { Received, From, To, (Cc), Subject, Text }
            for item in thread[docnum]:
                if item.tag == 'Subject':
                    print('\n    Email subject: "' + thread[docnum][3].text + '"')
                if item.tag == 'Text':
                    for sent in item:
                        print('        Sentence id: ' + sent.attrib['id'])
                        print('        Sentence: "' + sent.text + '"')
                        thread_text.append(sent.text)

        threads.append(thread_text)
        print('\n')

    sentence_vectors = calculate_features(threads)
    print(sentence_vectors)

xmlfile = open('baseline/data/corpus-tiny.xml', 'r')
parseXML(xmlfile)

xmlfile.close()
