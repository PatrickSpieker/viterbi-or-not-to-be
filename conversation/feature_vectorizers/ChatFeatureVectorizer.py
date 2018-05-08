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

class ChatFeatureVectorizer(FeatureVectorizer):

    def tf_idf(self, input, thread_index, thread, sentence_index, sentence):
        return self.TF_IDF_FEATURES[thread_index]

    def tf_isf(self, input, thread_index, thread, sentence_index, sentence):
        return 0

    def sentence_length(self, input, thread_index, thread, sentence_index, sentence):
        return len(sentence)

    def sentence_position(self, input, thread_index, thread, sentence_index, sentence):
        return sentence_index
    
    def title_similarity(self, input, thread_index, thread, sentence_index, sentence):
        return 0

    def centroid_coherence(self, input, thread_index, thread, sentence_index, sentence):
        return 0

    def special_terms(self, input, thread_index, thread, sentence_index, sentence):
        return self.SENT_SPECIAL_COUNTS[thread_index][sentence_index] / self.THREAD_SPECIAL_COUNTS[thread_index] if self.THREAD_SPECIAL_COUNTS[thread_index] != 0 else 0

    def is_question(self, input, thread_index, thread, sentence_index, sentence):
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
        """number_count = 0
        sent_tokens = tokenize.word_tokenize(sentence)
        tagged_sent = tagger.pos_tag(sent_tokens)
        for word_index, tagged_word in enumerate(tagged_sent):
            pos = tagged_word[1]
            if pos == 'CD':
                number_count += 1
        return number_count"""
        return len(re.findall('\d', sentence))

    def url_count(self, input, thread_index, thread, sentence_index, sentence):
        return len(re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', sentence))
