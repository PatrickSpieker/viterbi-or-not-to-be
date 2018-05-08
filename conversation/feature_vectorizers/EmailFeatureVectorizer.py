from feature_vectorizers.FeatureVectorizer import FeatureVectorizer
import numpy as np
import re
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
        if sentence_index in self.TF_ISF_CACHE:
            tf_isf = self.TF_ISF_CACHE[sentence_index]
        else:
            thread_with_name = thread.copy()
            thread_with_name.append(input['names'][thread_index])
            tf_isf_vectorizer = TfidfVectorizer()
            tf_isf = tf_isf_vectorizer.fit_transform(thread_with_name)
            self.TF_ISF_CACHE[sentence_index] = tf_isf

        tf_isf_features = np.squeeze(np.asarray(np.mean(tf_isf, axis=1)))
        return tf_isf_features[sentence_index]

    def sentence_length(self, input, thread_index, thread, sentence_index, sentence):
        return len(sentence)

    def sentence_position(self, input, thread_index, thread, sentence_index, sentence):
        return sentence_index
    
    def similarity_to_title(self, input, thread_index, thread, sentence_index, sentence):
        if sentence_index in self.TF_ISF_CACHE:
            tf_isf = self.TF_ISF_CACHE[sentence_index]
        else:
            thread_with_name = thread.copy()
            thread_with_name.append(input['names'][thread_index])
            tf_isf_vectorizer = TfidfVectorizer()
            tf_isf = tf_isf_vectorizer.fit_transform(thread_with_name)
            self.TF_ISF_CACHE[sentence_index] = tf_isf

        title_vector = tf_isf[len(thread_with_name) - 1]
        sentence_vector = tf_isf[sentence_index]
        return linear_kernel(title_vector, sentence_vector).flatten()

    def centroid_coherence(self, input, thread_index, thread, sentence_index, sentence):
        if sentence_index in self.TF_ISF_CACHE:
            tf_isf = self.TF_ISF_CACHE[sentence_index]
        else:
            thread_with_name = thread.copy()
            thread_with_name.append(input['names'][thread_index])
            tf_isf_vectorizer = TfidfVectorizer()
            tf_isf = tf_isf_vectorizer.fit_transform(thread_with_name)
            self.TF_ISF_CACHE[sentence_index] = tf_isf

        tf_isf_mean = np.mean(tf_isf, axis=0)
        sentence_vector = tf_isf[sentence_index]
        return linear_kernel(tf_isf_mean, sentence_vector).flatten()

    def special_terms(self, input, thread_index, thread, sentence_index, sentence):
        return self.SENT_SPECIAL_COUNTS[thread_index][sentence_index] / self.THREAD_SPECIAL_COUNTS[thread_index] if self.THREAD_SPECIAL_COUNTS[thread_index] != 0 else 0

    """def is_question(self, input, thread_index, thread, sentence_index, sentence):
        return 1 if sentence.endswith('?') else 0

    def sentiment_score(self, input, thread_index, thread, sentence_index, sentence):
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

    def number_count(self, input, thread_index, thread, sentence_index, sentence):
        number_count = 0
        sent_tokens = tokenize.word_tokenize(sentence)
        tagged_sent = tagger.pos_tag(sent_tokens)
        for word_index, tagged_word in enumerate(tagged_sent):
            pos = tagged_word[1]
            if pos == 'CD':
                number_count += 1
        return number_count

    def url_count(self, input, thread_index, thread, sentence_index, sentence):
        return len(re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', sentence))"""
