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
        'title_similarity',
        'centroid_coherence',
        'special_terms',
        'is_question',
        'sentiment_score',
        'number_count',
        'url_count',
        'position_from_end'
    ]
    NUM_FEATURES = len(FEATURES)
    TF_IDF_FEATURES = []
    TF_ISF_CACHE = {}
    THREAD_SPECIAL_COUNTS = []
    SENT_SPECIAL_COUNTS = {}
    TOPIC_DIVISIONS = []

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
        paragraphs = ['\n\n'.join(x) for x in collapsed_threads]

        # Determine the number of sentences using the specific input format
        # for this data type
        num_sentences = 0
        for thread in threads:
            for chunk in thread:
                num_sentences += len(chunk)

        # Create an appropriately shaped array to hold the feature vectors
        sentence_features = np.ndarray(shape=(num_sentences, self.NUM_FEATURES))

        # Compute TF_IDF_FEATURES
        tf_idf_vectorizer = TfidfVectorizer()
        tf_idf = tf_idf_vectorizer.fit_transform(documents)
        self.TF_IDF_FEATURES = np.squeeze(np.asarray(np.mean(tf_idf, axis=1)), axis=1)

        # Compute topic divisions
        text_tiler = tokenize.TextTilingTokenizer(demo_mode=True)
        for topic in paragraphs:
            _, _, _, topic_boundaries = text_tiler.tokenize(topic)

            position_since_last_boundary = 0
            for sentence_index, is_boundary in enumerate(topic_boundaries):
                if is_boundary == 1:
                    position_since_last_boundary = 0
                else:
                    position_since_last_boundary += 1
                topic_boundaries[sentence_index] = position_since_last_boundary

            self.TOPIC_DIVISIONS.append(topic_boundaries)

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
        return sentence_index

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
        return total_score / len(tagged_sent)

    def number_count(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return len(re.findall('\d', sentence))

    def url_count(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return len(re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', sentence))

    def position_from_end(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return len(chunk) - sentence_index

    def topic_position(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return self.TOPIC_DIVISIONS[thread_index][thread_sentence_index]
