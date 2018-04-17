# Generate reference summaries from annotated data for evaluation

import configuration as config
import glob
import numpy as np
import xml.etree.ElementTree as ET
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import linear_kernel
from functools import reduce
from scipy import spatial

def main():
    with open(config.DATA_DIR + config.CORPUS + config.VALIDATION, 'r') as corpus_file, open(config.DATA_DIR + config.ANNOTATIONS + config.VALIDATION, 'r') as annotations_file: 
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
    output_dir = config.OUTPUT + config.REFERENCE

    if os.path.exists(output_dir):
        for f in glob.glob(output_dir + '*.txt'):
            os.remove(f)
    else:
        os.makedirs(output_dir)

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
                    if sentence_id in annotation:
                        summary.append(sent.text)

            filename = output_dir + 'thread{}_reference{}.txt'.format(thread_index, annotation_index)
            with open(filename, 'w') as output_file:
                output_file.write(''.join(summary))

if __name__ == '__main__':
    main()
