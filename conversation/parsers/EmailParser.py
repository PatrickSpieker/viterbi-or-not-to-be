import xml.etree.ElementTree as ET
import os
import glob

DEBUG = False

OUTPUT = 'output/'
REFERENCE = 'reference/'

class EmailParser:
    def __init__(self, overall_dir):
        self.overall_dir = overall_dir

    def corpus(self, partition):
        return '{}/{}/corpus.xml'.format(self.overall_dir, partition)

    def annotation(self, partition):
        return '{}/{}/annotation.xml'.format(self.overall_dir, partition)

    def parse(self, partition):
        threads, thread_labels, thread_names = self.load_data(self.corpus(partition), self.annotation(partition))

        data = {
            'data': threads,
            'labels': thread_labels,
            'names': thread_names
        }

        return data
        
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

                # TODO: triple-nesting threads and labels could allow more features

                for sent in text:
                    for annotation in annotations[listno]:
                        debug('        Sentence id: ' + sent.attrib['id'])
                        sentence_id = sent.attrib['id']
                        doc_labels.append(1 if sentence_id in annotation else 0)
                        debug('        Sentence: "' + sent.text + '"')
                        thread_text.append(sent.text)

            debug('\n')
            threads.append(thread_text)
            thread_labels.append(doc_labels)
            thread_names.append(name)

        return threads, thread_labels, thread_names

    def compile_reference_summaries(self):
        with open(self.corpus('val')) as corpus_file, open(self.annotation('val')) as annotations_file:
            annotations = self.parse_annotations(annotations_file)

            output_dir = OUTPUT + REFERENCE

            if os.path.exists(output_dir):
                for f in glob.glob(output_dir + '*'):
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
                                summary.append(sent.text + ' ')

                    filename = output_dir + 'thread{}_reference{}.txt'.format(thread_index, annotation_index)
                    with open(filename, 'w') as output_file:
                        output_file.write(''.join(summary))

def flatten(nested_list):
    return [label for thread in nested_list for label in thread]
        
def debug(output):
    if DEBUG:
        print(output)
