import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureVectorizer:

    NUM_FEATURES = 11
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
    TF_IDF_FEATURES = []
    TF_ISF_CACHE = {}

    def vectorize(self, input):
        """
        Vectorizes the given input according to the features for the specific
        feature-vectorizer.

        Returns an n-dimensional array of shape (NUM_SENTENCES, NUM_FEATURES).
        """

        threads = input['data']
        documents = [' '.join(x) for x in threads]

        # Determine the number of sentences using the specific input format
        # for this data type
        num_sentences = 0
        for thread in threads:
            num_sentences += len(thread)

        # Create an appropriately shaped array to hold the feature vectors
        sentence_features = np.ndarray(shape=(num_sentences, self.NUM_FEATURES))

        # Compute TF_IDF_FEATURES
        tf_idf_vectorizer = TfidfVectorizer()
        tf_idf = tf_idf_vectorizer.fit_transform(documents)
        self.TF_IDF_FEATURES = np.squeeze(np.asarray(np.mean(tf_idf, axis=1)), axis=1)

        # Populate the feature vector
        global_sentence_index = 0
        for thread_index, thread in enumerate(threads):
            for sentence_index, sentence in enumerate(thread):
                for feature_index, feature in enumerate(self.FEATURES):
                    feature_result = getattr(self, feature)(input, thread_index, thread, sentence_index, sentence)
                    sentence_features[global_sentence_index, feature_index] = feature_result
                global_sentence_index += 1

        return sentence_features

    # --- For subclasses, override these methods: ---

    def tf_idf(self, input, thread_index, thread, sentence_index, sentence):
        return 0

    def tf_isf(self, input, thread_index, thread, sentence_index, sentence):
        return 0

    def sentence_length(self, input, thread_index, thread, sentence_index, sentence):
        return 0

    def sentence_position(self, input, thread_index, thread, sentence_index, sentence):
        return 0

    def title_similarity(self, input, thread_index, thread, sentence_index, sentence):
        return 0

    def centroid_coherence(self, input, thread_index, thread, sentence_index, sentence):
        return 0

    def special_terms(self, input, thread_index, thread, sentence_index, sentence):
        return 0

    def is_question(self, input, thread_index, thread, sentence_index, sentence):
        return 0

    def sentiment_score(self, input, thread_index, thread, sentence_index, sentence):
        return 0

    def number_count(self, input, thread_index, thread, sentence_index, sentence):
        return 0

    def url_count(self, input, thread_index, thread, sentence_index, sentence):
        return 0
