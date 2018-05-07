from feature_vectorizers.FeatureVectorizer import FeatureVectorizer

class EmailFeatureVectorizer(FeatureVectorizer):

    def sentence_length(self, input, thread_index, thread, sentence_index, sentence):
        return len(sentence)
