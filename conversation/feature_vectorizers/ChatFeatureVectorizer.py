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
from nltk.translate.bleu_score import sentence_bleu
from rake_nltk import Rake

class ChatFeatureVectorizer(FeatureVectorizer):

    def title_similarity(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        return 0
        # r = Rake()
        # chunks = [' '.join(x) for x in thread]
        # threads = [' '.join(x) for x in chunks]
        # keywords = r.extract_keywords_from_text(threads[thread_index])
        # ranked_keywords = r.get_ranked_phrases()
        # title = ''
        # for i in range(0, min(5, len(ranked_keywords))):
        #     title += ranked_keywords[i] + ' '
        
        # return sentence_bleu(title, sentence)

    def topic_position(self, input, thread_index, thread, chunk_index, chunk, sentence_index, sentence, thread_sentence_index):
        #return 0
        try:
            return self.TOPIC_DIVISIONS[thread_index][thread_sentence_index]
        except IndexError:
            return 1