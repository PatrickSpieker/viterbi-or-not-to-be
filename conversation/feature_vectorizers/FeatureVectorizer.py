import numpy as np

class FeatureVectorizer:

    NUM_FEATURES = 9
    FEATURES = [
        'tf_idf',
        'tf_isf',
        'sentence_length',
        'sentence_position',
        'title_similarity',
        'centroid_coherence',
        'special_terms',
        'quoted_text',
        'position_from_end'
    ]

    def vectorize(self, input):
        """
        Vectorizes the given input according to the features for the specific
        feature-vectorizer.

        Returns an n-dimensional array of shape (NUM_SENTENCES, NUM_FEATURES).
        """

        threads = input['data']

        # Determine the number of sentences using the specific input format
        # for this data type
        num_sentences = 0
        for thread in threads:
            num_sentences += len(thread)

        # Create an appropriately shaped array to hold the feature vectors
        sentence_features = np.ndarray(shape=(num_sentences, self.NUM_FEATURES))

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

    def quoted_text(self, input, thread_index, thread, sentence_index, sentence):
        return 0
    
    def position_from_end(self, input, thread_index, thread, sentence_index, sentence):
        return 0