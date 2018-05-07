from feature_vectorizers.FeatureVectorizer import FeatureVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import linear_kernel
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from functools import reduce
from scipy import spatial
from nltk import tokenize
from nltk import tag as tagger
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn

class EmailFeatureVectorizer(FeatureVectorizer):

    def tf_idf(self, input, thread_index, thread, sentence_index, sentence):
        return self.TF_IDF_FEATURES[thread_index]

    def tf_isf(self, input, thread_index, thread, sentence_index, sentence):
        thread_with_name = thread.copy()
        thread_with_name.append(input['names'][thread_index])
        tf_isf_vectorizer = TfidfVectorizer()
        tf_isf = tf_isf_vectorizer.fit_transform(thread_with_name)
        tf_isf_features = np.squeeze(np.asarray(np.mean(tf_isf, axis=1)))
        return tf_isf_features[sentence_index]

    def sentence_length(self, input, thread_index, thread, sentence_index, sentence):
        return len(sentence)

    def sentence_position(self, input, thread_index, thread, sentence_index, sentence):
        return sentence_index
    
    def similarity_to_title(self, input, thread_index, thread, sentence_index, sentence):
        thread_with_name = thread.copy()
        thread_with_name.append(input['names'][thread_index])
        tf_isf_vectorizer = TfidfVectorizer()
        tf_isf = tf_isf_vectorizer.fit_transform(thread_with_name)

        title_vector = tf_isf[len(thread_with_name) - 1]
        sentence_vector = tf_isf[sentence_index]
        return linear_kernel(title_vector, sentence_vector).flatten()

    def centroid_coherence(self, input, thread_index, thread, sentence_index, sentence):
        thread_with_name = thread.copy()
        thread_with_name.append(input['names'][thread_index])
        tf_isf_vectorizer = TfidfVectorizer()
        tf_isf = tf_isf_vectorizer.fit_transform(thread_with_name)

        tf_isf_mean = np.mean(tf_isf, axis=0)
        sentence_vector = tf_isf[sentence_index]
        return linear_kernel(tf_isf_mean, sentence_vector).flatten()

    def special_terms(self, input, thread_index, thread, sentence_index, sentence):
        special_counts = []
        total_special_count = 0.0
        for sentence in thread:
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
            
            special_counts.append(sent_special_count)
            total_special_count = total_special_count + sent_special_count

        return special_counts[sentence_index] / total_special_count if total_special_count != 0 else 0
