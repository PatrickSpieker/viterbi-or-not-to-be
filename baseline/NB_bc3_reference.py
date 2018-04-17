
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import linear_kernel
from functools import reduce
from scipy import spatial

DATA_DIR = 'data/'

CORPUS = 'corpus-tiny'
ANNOTATIONS = 'annotations-tiny'

TRAIN = '.train.xml'
VALIDATION = '.val.xml'
TEST = '.test.xml'

OUTPUT = 'output/reference/'

def main():
    with open(DATA_DIR + CORPUS + VALIDATION, 'r') as corpus_file, open(DATA_DIR + ANNOTATIONS + VALIDATION, 'r') as annotations_file: 
        annotations = parse_annotations(annotations_file)
        output_summaries(corpus_file, annotations)
                
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
            annotation_map[listno].append(sentence_ids)

    return annotation_map

def output_summaries(corpus_file, annotations):
    tree = ET.parse(corpus_file)
    root = tree.getroot()

    for thread_index, thread in enumerate(root):
        listno = thread.find('listno').text
        annotations_list = annotations[listno]
        
        for annotation_index, annotation in enumerate(annotations_list):
            summary = []
            
            for doc in thread.findall('DOC'):
                text = doc.find('Text')
                for sent in text:
                    sentence_id = sent.attrib['id']
                    if sentence_id in annotations[listno]:
                        summary.append(sent.text)

            filename = OUTPUT + 'thread{}_reference{}.txt'.format(thread_index, annotation_index)
            with open(filename, 'w+') as output_file:
                output_file.write(' '.join(summary))

if __name__ == '__main__':
    main()
