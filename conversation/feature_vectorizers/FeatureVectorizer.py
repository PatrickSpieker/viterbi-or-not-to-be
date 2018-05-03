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

    def vectorize(self, *input):
        """Returns a (FEATURES) shape feature vector"""
        feature_vector = map(lambda x : getattr(self, x)(input), self.FEATURES)
        return feature_vector

    # --- For subclasses, override these methods: ---
        
    def tf_idf(self, input):
        return 0

    def tf_isf(self, input):
        return 0

    def sentence_length(self, input):
        return 0

    def sentence_position(self, input):
        return 0

    def title_similarity(self, input):
        return 0

    def centroid_coherence(self, input):
        return 0

    def special_terms(self, input):
        return 0

    def quoted_text(self, input):
        return 0
    
    def position_from_end(self, input):
        return 0