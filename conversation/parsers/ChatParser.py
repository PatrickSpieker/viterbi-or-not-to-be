import xml.etree.ElementTree as ET
import os
import glob

# The subdirectories under which ROUGE-compatible summaries should be output
OUTPUT = 'output/'
REFERENCE = 'reference/'

class ChatParser:
    def __init__(self, overall_dir, debug):
        self.overall_dir = overall_dir
        self.debug_flag = debug

    def corpus(self, partition):
        return '{}/{}/corpus'.format(self.overall_dir, partition)

    def annotation(self, partition):
        return '{}/{}/annotation'.format(self.overall_dir, partition)

    def parse(self, partition):
        threads, thread_labels, thread_names = self.load_data(self.corpus(partition), self.annotation(partition))

        data = {
            'data': threads,
            'labels': thread_labels,
            'names': thread_names
        }

        return data
        
    def load_data(self, corpus_dir, annotation_dir):
        threads, thread_labels, thread_names = self.parse_corpus_and_annotations(corpus_dir, annotation_dir)
        return threads, thread_labels, thread_names

    def parse_corpus_and_annotations(self, corpus_dir, annotation_dir):
        threads = []
        thread_labels = []
        thread_names = []

        for anno_filename in os.listdir(annotation_dir):
            curr_thread = []
            curr_thread_labels = []
            quotes = set()
            anno_file = os.path.join(annotation_dir, anno_filename)
            thread_index = anno_filename.split('-')[1].split('.')[0]
            tree = ET.parse(anno_file)
            root = tree.getroot()
            for p in root.findall('p'):
                for q in p.findall('quote'):
                    quotes.add(q.text.replace('\n', ''))

            corpus_filename = 'corpus-' + thread_index + '.txt'
            corpus_file = open(os.path.join(corpus_dir, corpus_filename), 'r', errors='ignore')
            for line in corpus_file.readlines():
                line = line.replace('\n', '')
                curr_thread.append(line)
                for q in quotes:
                    if q.replace(',', '') in line.replace(',', ''):
                        curr_thread_labels.append(1)
                    else:
                        curr_thread_labels.append(0)
            nested_thread = []
            nested_labels = []
            nested_thread.append(curr_thread)
            nested_labels.append(curr_thread_labels)
            threads.append(nested_thread)
            thread_labels.append(nested_labels)
        return threads, thread_labels, thread_names

    def compile_reference_summaries(self):
        for anno_filename in os.listdir(self.annotation('val')):
            anno_file = os.path.join(self.annotation('val'), anno_filename)
            thread_index = anno_filename.split('-')[1].split('.')[0]
            output_dir = OUTPUT + REFERENCE

            if os.path.exists(output_dir):
                for f in glob.glob(output_dir + '*'):
                    os.remove(f)
            else:
                os.makedirs(output_dir)

            tree = ET.parse(anno_file)
            root = tree.getroot()

            filename = output_dir + 'thread{}.txt'.format(thread_index)
            with open(filename, 'w') as output_file:
                for p in root.findall('p'):
                    output_file.write(p.text)

    def debug(self, output):
        if self.debug_flag:
            print(output)

def flatten(nested_list):
    return [label for thread in nested_list for label in thread]
