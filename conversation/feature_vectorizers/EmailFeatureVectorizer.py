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

    """def title_similarity(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence):
        if sentence_index in self.TF_ISF_CACHE:
            tf_isf = self.TF_ISF_CACHE[sentence_index]
        else:
            #thread_with_name = thread.copy()
            thread_with_name = self.flatten(thread.copy())
            thread_with_name.append(input['names'][thread_index])
            tf_isf_vectorizer = TfidfVectorizer()
            tf_isf = tf_isf_vectorizer.fit_transform(thread_with_name)
            self.TF_ISF_CACHE[sentence_index] = tf_isf
        
        title_vector = tf_isf[tf_isf.shape[0] - 1]
        sentence_vector = tf_isf[sentence_index]
        return linear_kernel(title_vector, sentence_vector).flatten()"""
