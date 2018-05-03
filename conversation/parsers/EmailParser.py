import xml.etree.ElementTree as ET
import os

DEBUG = False

class EmailParser:
    def __init__(self, overall_dir, partition):
        self.corpus_file = '{}/{}/corpus.xml'.format(overall_dir, partition)
        self.annotations_file = '{}/{}/annotation.xml'.format(overall_dir, partition)

    # TODO: emails should be in their own partitions in the list
    def parse(self):
        """
        Should have consistent API
        """
        threads, thread_labels, thread_names = self.load_data(self.corpus_file, self.annotations_file)
        preprocessed_data = self.preprocess(threads, thread_labels, thread_names)
        return preprocessed_data
    
    def load_data(self, corpus_file, annotations_file):
        annotations = self.parse_annotations(annotations_file)
        threads, thread_labels, thread_names = self.parse_corpus(corpus_file, annotations)
        return threads, thread_labels, thread_names

    def parse_annotations(self, xml_file):
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

    def parse_corpus(self, xml_file, annotations):
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

    def preprocess(self, threads, thread_labels, thread_names):
        return threads, thread_labels, thread_names

def flatten(nested_list):
    return [label for thread in nested_list for label in thread]
        
def debug(output):
    if DEBUG:
        print(output)
