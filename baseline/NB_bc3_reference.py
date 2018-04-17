
import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
import os
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

    if os.path.exists('output/reference'):
        for f in glob.glob('output/reference/*.txt'):
            os.remove(f)
    else:
        os.makedirs('output/reference')

    if os.path.exists('output/system'):
        for f in glob.glob('output/system/*.txt'):
            os.remove(f)
    else:
        os.makedirs('output/system')

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
<<<<<<< HEAD
            with open(filename, 'w') as output_file:
=======
            os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
            with open(filename, 'w+') as output_file:
>>>>>>> 6a93f450e29b7402770b19364bddda4edc720eca
                output_file.write(' '.join(summary))

if __name__ == '__main__':
    main()
