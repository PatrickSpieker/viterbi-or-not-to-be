import re

class ChatPreprocessor:

    SENTENCE_END = set(['.', '!', '?'])

    def preprocess(self, input):
        return self.longest_contiguous_message(input)

    def longest_contiguous_message(self, input):
        result = {
            'data': [],
            'labels': [],
            'names': [],
            'authors': [],
            'flat_authors': []
        }

        username = re.compile('^<(.*)> (.*)$')

        for thread_index, thread in enumerate(input['data']):
            # Each list contains sublists, with each sublist representing
            # a single "new" chunk in this thread
            new_thread_text = []
            new_thread_labels = []
            new_thread_authors = []
            new_thread_flat_authors = []

            # Variables used to store current state of chunk division algorithm
            # at any point
            current_username = None
            new_chunk_text = []
            new_chunk_labels = []

            for chunk_index, chunk in enumerate(thread):
                # No work is done in this dimension because it is being
                # overwritten; it will be rearranged into new chunks

                for sentence_index, sentence in enumerate(chunk):
                    # If there is no username, it is not
                    # a part of the conversation and should
                    # be dropped.
                    sentence_match = username.match(sentence)

                    # Only process sentences with usernames and with message bodies
                    if sentence_match is not None and sentence_match.group(2).strip() != '':
                        if current_username is None:
                            # If this is the first sentence in a chunk, store
                            # the username as the first username
                            current_username = sentence_match.group(1)

                        elif current_username != sentence_match.group(1):
                            # It is time to create a new chunk because the username
                            # has changed. To do so, save the current chunk and
                            # prepare a new chunk to work on under new_chunk_text.
                            if new_chunk_text[-1][-1] not in self.SENTENCE_END:
                                new_chunk_text[-1] += '.'
                            new_thread_text.append(new_chunk_text)
                            new_thread_labels.append(new_chunk_labels)
                            new_thread_authors.append(current_username)
                            for i in range(len(new_chunk_text)):
                                new_thread_flat_authors.append(current_username)

                            new_chunk_text = []
                            new_chunk_labels = []
                            current_username = sentence_match.group(1)

                        # At this point, new_chunk_text refers to the correct
                        # chunk to be modifying. Add the sentence, ending with
                        # a period if it does not already.
                        new_sentence = sentence_match.group(2)

                        new_chunk_text.append(new_sentence)
                        new_chunk_labels.append(input['labels'][thread_index][chunk_index][sentence_index])

            new_thread_text.append(new_chunk_text)
            new_thread_labels.append(new_chunk_labels)
            new_thread_authors.append(current_username)
            for i in range(len(new_chunk_text)):
                new_thread_flat_authors.append(current_username)

            # The chunks for this thread have been computed
            result['data'].append(new_thread_text)
            result['labels'].append(new_thread_labels)
            result['authors'].append(new_thread_authors)
            result['flat_authors'].append(new_thread_flat_authors)
        
        return result