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

        # Try to determine the boundary of an email signature and remove it
        for thread_index, thread in enumerate(processed_threads_text):
            for email_index, email in enumerate(thread):
                last_content = len(email) - 1

                # Iterate backwards through the email, searching for the last line likely to contain content
                while self.signature(email, last_content):
                    last_content -= 1

                thread[email_index] = email[:last_content + 1]
                processed_threads_labels[thread_index][email_index] = processed_threads_labels[thread_index][email_index][:last_content + 1]

        # Return the preprocesed input
        processed_input['data'] = processed_threads_text
        processed_input['labels'] = processed_threads_labels

        return processed_input

    def signature(self, email, line_index):
        if line_index <= 0:
            return False
        line = email[line_index].strip()
        return len(line) > 0 and \
               (line[-1] not in ['!', '?', '.', '-', '\'', '"', ')', ':', ']', '>', '}', '/']) and \
               (not re.match(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', line))

    def quoted_text(self, sentence):
        return sentence.strip()[0] == '>' or sentence.strip()[:4] == '&gt;'

    def only_symbols(self, sentence):
        return not re.search('[a-zA-Z0-9]', sentence)