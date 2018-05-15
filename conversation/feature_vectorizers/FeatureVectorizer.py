import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import tokenize
from nltk import tag as tagger
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
        'url_count'
    ]
    NUM_FEATURES = len(FEATURES)
    TF_IDF_FEATURES = []
    TF_ISF_CACHE = {}
    THREAD_SPECIAL_COUNTS = []
    SENT_SPECIAL_COUNTS = {}

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
                for chunk_index, chunk in enumerate(thread):
                    for sentence_index, sentence in enumerate(chunk):
                        for feature_index, feature in enumerate(self.FEATURES):
                            feature_result = getattr(self, feature)(input, thread_index, thread, chunk_index, chunk, sentence_index, sentence)
                            sentence_features[global_sentence_index, feature_index] = feature_result
                        global_sentence_index += 1
                    TF_ISF_CACHE = {}
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

    def tf_idf(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence):
        return 0

    def tf_isf(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence):
        return 0

    def sentence_length(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence):
        return 0

    def sentence_position(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence):
        return 0

    def title_similarity(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence):
        return 0

    def centroid_coherence(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence):
        return 0

    def special_terms(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence):
        return 0

    def is_question(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence):
        return 0

    def sentiment_score(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence):
        return 0

    def number_count(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence):
        return 0

    def url_count(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence):
        return 0
