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

    def tf_isf(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence):
        return 0

    def title_similarity(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence):
        return 0

    def centroid_coherence(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence):
        return 0
