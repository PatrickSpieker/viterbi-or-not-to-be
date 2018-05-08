import re
import pdb

class EmailPreprocessor:
    def preprocess(self, input):
        processed_input = {
            'names': input['names'],
        }

        # Remove quoted text and any "sentences" that are composed of solely
        # symbols.
        processed_threads_text = []
        processed_threads_labels = []
        for thread_index, thread in enumerate(input['data']):
            processed_thread_text = []
            processed_thread_labels = []
            for email_index, email in enumerate(thread):
                processed_email_text = []
                processed_email_labels = []
                for sentence_index, sentence in enumerate(email):
                    if not (self.quoted_text(sentence) or self.only_symbols(sentence)):
                        processed_email_text.append(sentence)
                        processed_email_labels.append(input['labels'][thread_index][email_index][sentence_index])
                processed_thread_text.append(processed_email_text)
                processed_thread_labels.append(processed_email_labels)
            processed_threads_text.append(processed_thread_text)
            processed_threads_labels.append(processed_thread_labels)

        # Try to determine the boundary of an email
        dog = 'lol no'
                    
        # Return the preprocesed input
        processed_input['data'] = processed_threads_text
        processed_input['labels'] = processed_threads_labels

        return processed_input

    def quoted_text(self, sentence):
        return sentence.strip()[0] == '>' or sentence.strip()[:4] == '&gt;'

    def only_symbols(self, sentence):
        return not re.search('[a-zA-Z0-9]', sentence)