import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk import tokenize
from nltk import tag as tagger
from nltk.corpus import sentiwordnet as swn
import re
np.set_printoptions(threshold=np.inf)

class FeatureVectorizer:

    FEATURES = [
        'tf_idf',
        'tf_isf',
        'sentence_length',
        'sentence_position',
        'thread_sentence_position',
        'title_similarity',
        'centroid_coherence',
        'special_terms',
        'is_question',
        'sentiment_score',
        'number_count',
        'url_count',
        'position_from_end',
        'topic_position',
        'previous_tf_isf',
        'author_frequency',
        'result_relation',
        'circumstance_relation'
    ]
    NUM_FEATURES = len(FEATURES)
    TF_IDF_FEATURES = []
    TF_ISF_CACHE = {}
    THREAD_SPECIAL_COUNTS = []
    SENT_SPECIAL_COUNTS = {}
    TOPIC_DIVISIONS = []
    THREAD_AUTHOR_COUNTS = []
    THREAD_SENTENCE_COUNTS = []
    RESULT_RELATION_MARKERS = ['because of', 'as a result of', 'because', 'and', 'so', 'as a result', 'when', 
        'as', 'since', 'now', 'after', 'the result', 'so far', 'now that', 'and so', 'thus', 'but']
    CIRCUMSTANCE_RELATION_MARKERS = ['when', 'as', 'after', 'following', 'since', 'and', 'without', 'but', 
        'once', 'until', 'with', 'before', 'now', 'while', 'if', 'given', 'because']

    def vectorize(self, input):
        """
        Vectorizes the given input according to the features for the specific
        feature-vectorizer.

        Returns an n-dimensional array of shape (NUM_SENTENCES, NUM_FEATURES).
        """

        print('Vectorizing: Computing Special Features')

        threads = input['data']
        collapsed_threads = self.collapse_threads(threads)
        documents = [' '.join(x) for x in collapsed_threads]

        paragraphs = []
        for thread in collapsed_threads:
            cleaned_sentences = [x.replace('\n', '') for x in thread]
            paragraphs.append('\n\n'.join(cleaned_sentences))
                
        # Determine the number of sentences using the specific input format
        # for this data type and build up author counts
        num_sentences = 0
        authors = input['authors']
        for thread_index, thread in enumerate(threads):
            thread_num_sentences = 0
            author_counts = {}
            for chunk_index, chunk in enumerate(thread):
                chunk_author = authors[thread_index][chunk_index]
                if chunk_author in author_counts:
                    author_counts[chunk_author] += 1
                else:
                    author_counts[chunk_author] = 1
                num_sentences += len(chunk)
                thread_num_sentences += len(chunk)
            self.THREAD_AUTHOR_COUNTS.append(author_counts)
            self.THREAD_SENTENCE_COUNTS.append(thread_num_sentences)

        # Create an appropriately shaped array to hold the feature vectors
        sentence_features = np.ndarray(shape=(num_sentences, self.NUM_FEATURES))

        # Compute TF_IDF_FEATURES
        tf_idf_vectorizer = TfidfVectorizer()
        tf_idf = tf_idf_vectorizer.fit_transform(documents)
        self.TF_IDF_FEATURES = np.squeeze(np.asarray(np.mean(tf_idf, axis=1)), axis=1)

        # Compute topic divisions
        # text_tiler = tokenize.TextTilingTokenizer(demo_mode=False)
        # for thread in paragraphs:
        #     topic_boundaries = text_tiler.tokenize(thread)
        #     thread_positions = []
        #     for topic_boundary in topic_boundaries:
        #         sentences = topic_boundary.split('\n\n')
        #         if len(sentences[0]) == 0:
        #             sentences = sentences[1:]

        #         num_topic_sentences = len(sentences)
        #         for sentence_index, sentence in enumerate(sentences):
        #             thread_positions.append(sentence_index / num_topic_sentences)

        #     self.TOPIC_DIVISIONS.append(thread_positions)

        # Count special terms per sentence, thread
        with tqdm(total=len(collapsed_threads)) as pbar:
            for thread_index, collapsed_threads in enumerate(collapsed_threads):
                thread_special_count = 0
                self.SENT_SPECIAL_COUNTS[thread_index] = []
                for sentence_index, sentence in enumerate(collapsed_threads):
                    sent_tokens = tokenize.word_tokenize(sentence)
                    tagged_sent = tagger.pos_tag(sent_tokens)
                    prev_proper_index = -10
                    sent_special_count = 0.0

                    for word_index, tagged_word in enumerate(tagged_sent):
                        pos = tagged_word[1]
                        if pos == 'NNP':
                            if prev_proper_index != word_index - 1:
                                sent_special_count += 1.0
                            prev_proper_index = word_index
                        elif pos == 'CD':
                            sent_special_count += 1.0
                    
                    self.SENT_SPECIAL_COUNTS[thread_index].append(sent_special_count)
                    thread_special_count += sent_special_count
                self.THREAD_SPECIAL_COUNTS.append(thread_special_count)
                pbar.update(1)

        print('Vectorizing: Computing Features')

        # Populate the feature vector
        global_sentence_index = 0
        with tqdm(total=len(threads)) as pbar:
            for thread_index, thread in enumerate(threads):
                thread_sentence_index = 0
                for chunk_index, chunk in enumerate(thread):
                    for sentence_index, sentence in enumerate(chunk):
                        for feature_index, feature in enumerate(self.FEATURES):
                            feature_result = getattr(self, feature)(input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index)
                            sentence_features[global_sentence_index, feature_index] = feature_result
                        global_sentence_index += 1
                        thread_sentence_index += 1
                    self.TF_ISF_CACHE = {}
                pbar.update(1)

        return sentence_features

    def flatten(self, nested_list):
        return [label for thread in nested_list for label in thread]

    def collapse_threads(self, threads):
        collapsed_threads = []
        for thread in threads:
            collapsed_threads.append(self.flatten(thread))
        return collapsed_threads
    
    # --- For subclasses, override these methods: ---

    def tf_idf(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return self.TF_IDF_FEATURES[thread_index]

    def tf_isf(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        if chunk_index in self.TF_ISF_CACHE:
            tf_isf = self.TF_ISF_CACHE[chunk_index]
        else:
            chunks = [' '.join(x) for x in thread]
            tf_isf_vectorizer = TfidfVectorizer()
            tf_isf = tf_isf_vectorizer.fit_transform(chunks)
            self.TF_ISF_CACHE[chunk_index] = tf_isf

        tf_isf_features = np.squeeze(np.asarray(np.mean(tf_isf, axis=1)))
        return tf_isf_features[chunk_index]

    def sentence_length(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return len(sentence)

    def sentence_position(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return sentence_index / len(chunk)

    def thread_sentence_position(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return thread_sentence_index / len(self.collapse_threads(thread))

    def title_similarity(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return 0

    def centroid_coherence(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        if chunk_index in self.TF_ISF_CACHE:
            tf_isf = self.TF_ISF_CACHE[chunk_index]
        else:
            chunks = [' '.join(x) for x in thread]
            tf_isf_vectorizer = TfidfVectorizer()
            tf_isf = tf_isf_vectorizer.fit_transform(chunks)
            self.TF_ISF_CACHE[chunk_index] = tf_isf

        tf_isf_mean = np.mean(tf_isf, axis=0)
        sentence_vector = tf_isf[chunk_index]
        return linear_kernel(tf_isf_mean, sentence_vector).flatten()

    def special_terms(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return self.SENT_SPECIAL_COUNTS[thread_index][sentence_index] / self.THREAD_SPECIAL_COUNTS[thread_index] if self.THREAD_SPECIAL_COUNTS[thread_index] != 0 else 0

    def is_question(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return 1 if sentence.endswith('?') else 0

    def sentiment_score(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        tag_set = {'NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
        total_score = 0.0
        sent_tokens = tokenize.word_tokenize(sentence)
        tagged_sent = tagger.pos_tag(sent_tokens)
        for word_index, tagged_word in enumerate(tagged_sent):
            tag = tagged_word[1]
            if tag in tag_set:
                pos = {
                    'NN' : 'n',
                    'NNS' : 'n',
                    'VB' : 'v',
                    'VBD' : 'v',
                    'VBG' : 'v',
                    'VBN' : 'v',
                    'VBP' : 'v',
                    'VBZ' : 'v',
                    'JJ' : 'a',
                    'JJR' : 'a',
                    'JJS' : 'a',
                    'RB' : 'r',
                    'RBR' : 'r',
                    'RBS' : 'r'
                }[tag]
                senti_str = tagged_word[0].lower() + '.' + pos + '.01'
                try:
                    senti_set = swn.senti_synset(senti_str)
                    senti_score = senti_set.pos_score() - senti_set.neg_score()
                    total_score += senti_score
                except:
                    pass
        if len(tagged_sent) == 0:
            return 0
        else:
            return total_score / len(tagged_sent)

    def number_count(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return len(re.findall('\d', sentence))

    def url_count(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return len(re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', sentence))

    def position_from_end(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return (len(chunk) - sentence_index) / len(chunk)

    def topic_position(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return 0

    def previous_tf_isf(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        previous_index = chunk_index - 1
        if chunk_index == 0:
            previous_index = chunk_index

        if previous_index in self.TF_ISF_CACHE:
            tf_isf = self.TF_ISF_CACHE[previous_index]
        else:
            chunks = [' '.join(x) for x in thread]
            tf_isf_vectorizer = TfidfVectorizer()
            tf_isf = tf_isf_vectorizer.fit_transform(chunks)
            self.TF_ISF_CACHE[previous_index] = tf_isf

        tf_isf_features = np.squeeze(np.asarray(np.mean(tf_isf, axis=1)))
        return tf_isf_features[previous_index]

    def author_frequency(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        chunk_author = input['authors'][thread_index][chunk_index]
        if chunk_author not in self.THREAD_AUTHOR_COUNTS[thread_index]:
            return 0
        return self.THREAD_AUTHOR_COUNTS[thread_index][chunk_author] / self.THREAD_SENTENCE_COUNTS[thread_index]

    def result_relation(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return 1 if sentence.lower().startswith(tuple(self.RESULT_RELATION_MARKERS)) else 0

    def circumstance_relation(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return 1 if sentence.lower().startswith(tuple(self.CIRCUMSTANCE_RELATION_MARKERS)) else 0
