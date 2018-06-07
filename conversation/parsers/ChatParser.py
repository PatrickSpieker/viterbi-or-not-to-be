import xml.etree.ElementTree as ET
from nltk.translate.bleu_score import sentence_bleu
import os
import glob
import re
from tqdm import tqdm

# The subdirectories under which ROUGE-compatible summaries should be output
OUTPUT = 'output/'
REFERENCE = 'reference/'

SIMILARITY = 0.01

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

        print('Parsing')
        files = glob.glob('{}/*.txt'.format(annotation_dir))

        with tqdm(total=len(files)) as pbar:
            for anno_filename in files:
                curr_thread = []
                curr_thread_labels = []
                quotes = []
                anno_file = anno_filename # os.path.join(annotation_dir, anno_filename)
                thread_index = anno_filename.split('-')[1].split('.')[0]
                tree = ET.parse(anno_file)
                root = tree.getroot()
                for p in root.findall('p'):
                    p_str = ET.tostring(p).decode("utf-8")
                    p_str = re.sub('<p>', '', p_str)
                    p_str = re.sub('</p>', '', p_str)
                    p_str = re.sub(r'^<quote(.*)>$', '', p_str)
                    p_str = re.sub('</quote>', '', p_str)
                    p_str = re.sub(r'^<a(.*)>$', '', p_str)
                    p_str = re.sub('</a>', '', p_str)
                    p_str_sents = p_str.split("\\.")
                    for sent in p_str_sents:
                        quotes.append(sent.replace('\n', '').replace(',', '').strip().split())
                    #for q in p.findall('quote'):
                    #    if q.text is not None:
                    #        quotes.append(q.text.replace('\n', '').replace(',', '').strip().split())

                corpus_filename = 'corpus-' + thread_index + '.txt'
                corpus_file = open(os.path.join(corpus_dir, corpus_filename), 'r', errors='ignore')
                for line in corpus_file.readlines():
                    line = line.replace('\n', '').replace(',', '').strip().split()
                    curr_thread.append(' '.join(line))
                    try:
                        similarity = sentence_bleu(quotes, line)
                        if similarity > SIMILARITY:
                            curr_thread_labels.append(1)
                        else:
                            curr_thread_labels.append(0)
                    except KeyError:
                        curr_thread_labels.append(0)
                
                nested_thread = []
                nested_labels = []
                nested_thread.append(curr_thread)
                nested_labels.append(curr_thread_labels)
                threads.append(nested_thread)
                thread_labels.append(nested_labels)
                pbar.update(1)

        return threads, thread_labels, thread_names

    def compile_reference_summaries(self):
        output_dir = OUTPUT + REFERENCE
        if os.path.exists(output_dir):
            for f in glob.glob(output_dir + '*'):
                os.remove(f)
        else:
            os.makedirs(output_dir)
            
        for anno_index, anno_filename in enumerate(os.listdir(self.annotation('val'))):
            anno_file = os.path.join(self.annotation('val'), anno_filename)
            try:
                thread_index = anno_filename.split('-')[1].split('.')[0]
            except:
                continue

            tree = ET.parse(anno_file)
            root = tree.getroot()

            filename = output_dir + 'thread{}_reference1.txt'.format(anno_index)
            with open(filename, 'w') as output_file:
                for p in root.findall('p'):
                    original_str = ET.tostring(p).decode("utf-8")
                    p_str = re.sub('<p>', '', original_str)
                    p_str = re.sub('</p>', '', p_str)
                    p_str = re.sub('<quote(.*?)>', '', p_str)
                    p_str = re.sub('</quote>', '', p_str)
                    p_str = re.sub('<a(.*?)>', '', p_str)
                    p_str = re.sub('</a>', '', p_str)
                    output_file.write(p_str) #.replace('\n', ''))
                    #for q in p.findall('quote'):
                    #    if q:
                    #        for p2 in q.findall('p'):
                    #            output_file.write(p2.text.replace('\n', '') + '\n')
                    #    else:
                    #        output_file.write(q.text.replace('\n', '') + '\n')

    def debug(self, output):
        if self.debug_flag:
            print(output)

def flatten(nested_list):
    return [label for thread in nested_list for label in thread]
