import re
import pdb

class EmailPreprocessor:
    def preprocess(self, input):
        processed_input = {
            'names': input['names'],
        }

        processed_threads = []
        processed_labels = []
        for thread_index, thread in enumerate(input['data']):
            processed_thread = []
            processed_label = []
            for sentence_index, sentence in enumerate(thread):
                if not (self.quoted_text(sentence) or self.only_symbols(sentence)):
                    processed_thread.append(sentence)
                    processed_label.append(input['labels'][thread_index][sentence_index])
            processed_threads.append(processed_thread)
            processed_labels.append(processed_label)
                    
        processed_input['data'] = processed_threads
        processed_input['labels'] = processed_labels

        return processed_input

    def quoted_text(self, sentence):
        return sentence.strip()[0] == '>'

    def only_symbols(self, sentence):
        return re.search('[a-zA-Z0-9]', sentence)