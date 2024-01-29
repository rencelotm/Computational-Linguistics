from nltk.probability import FreqDist, ConditionalFreqDist
from contextual_window import *
from math import log2


class PPMI:

    def __init__(self, words, context_size):
        self._co_occurrence = self.__construct_matrix_occurrence(words, context_size)
        self.__epsilon = .0001
        #self.__P_w_c = self.__compute_P_w_c(words)
        self.__sum = self.__total_sum(words)
        self.__P_c = self.__compute_P_c(words)
        self.__P_w = self.__compute_P_w(words)
        
    def get_k_closest_word(self, target, corpus, k):
        res = FreqDist()
        for word in set(corpus):
            res[word] += self.ppmi(target, word)
        return res.most_common(k)

    def ppmi(self, w_i, c_j):
        return max(self.__pmi(w_i, c_j), 0)

    def __construct_matrix_occurrence(self, corpus, context_size):    # This time co-occurrence matrix is matrix[word][context]
        context_constructor = ContextualWindowConstructor(context_size)
        matrix = ConditionalFreqDist()
        for i in range(len(corpus)):
            ctx_window = context_constructor.get_contextual_window(i, corpus).get_contextual_window_without_word()
            for j in range(len(ctx_window)):
                matrix[corpus[i]][ctx_window[j]] += 1
        return matrix

    def __pmi(self, w_i, c_j):
        return log2(((self._co_occurrence[w_i][c_j] + self.__epsilon) / self.__sum) / (self.__P_w[w_i] * self.__P_c[c_j]))
    
    def __total_sum(self, corpus):
        total_sum = 0
        for i in set(corpus):
            for j in set(corpus):
                total_sum += self._co_occurrence[i][j] + self.__epsilon
        return total_sum
    
    def __compute_P_w(self, corpus):
        res = FreqDist()
        for i in set(corpus):
            tmp = 0
            for j in set(corpus):
                tmp += ((self._co_occurrence[i][j] + self.__epsilon) / self.__sum)
            res[i] += tmp
        return res

    def __compute_P_c(self, corpus):
        res = FreqDist()
        for j in set(corpus):
            for i in set(corpus):
                res[j] += ((self._co_occurrence[i][j] + self.__epsilon) / self.__sum)
        return res
