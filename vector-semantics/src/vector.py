from contextual_window import *

class CoOccurrenceMatrix:

    def __init__(self, corpus, context_size):
        self.__w_ctx_constructor = ContextualWindowConstructor(context_size)
        self._co_occurrence = self.__construct_co_occurrence(corpus)
    
    def get_word_vector(self, word):
        return self._co_occurrence[word]
    
    def compute_cosine_similarity(self, v, w, words_set):
        numerator = 0
        v_i, w_i = 0, 0
        for word in words_set:
            numerator += v[word] * w[word]
            v_i += v[word] ** 2
            w_i += w[word] ** 2
        denominator = sqrt(v_i) * sqrt(w_i)
        return numerator / denominator
    
    def get_similarity_score(self, word, words_set):
        similarity_score = FreqDist()
        v_vect = self.get_word_vector(word)
        for w in words_set:
            if w != word:
                w_vect = self.get_word_vector(w)
                similarity_score[w] += self.compute_cosine_similarity(v_vect, w_vect, words_set)
        return similarity_score
    
    def get_k_closest_word(self, word, words_set, k):
        similarities = self.get_similarity_score(word, words_set)
        return dict(sorted(similarities.most_common(k), key=lambda x: (-x[1], x[0])))
    
    def __construct_co_occurrence(self, corpus):
        ret = ConditionalFreqDist()
        for i in range(len(corpus)):
            context_window = self.__w_ctx_constructor.get_contextual_window(i, corpus)
            for ctx_word in context_window.get_contextual_window_without_word():
                ret[corpus[i]][ctx_word] += 1
        return ret


class TFIDF(CoOccurrenceMatrix):

    def __init__(self, corpus, context_size):
        super().__init__(corpus, context_size)

    def __tf_idf(self, words_set, w_vect):
        tf_idf = FreqDist()
        idf = log10(len(words_set) / len(w_vect))
        for w in words_set:
            tf = log10(w_vect[w] + 1)
            tf_idf[w] += tf * idf
        return tf_idf
    
    def get_similarity_score(self, word, words_set):
        similarity_score = FreqDist()
        v_vect = self.__tf_idf(words_set, self.get_word_vector(word))
        for w in words_set:
            if w != word:
                w_vect = self.__tf_idf(words_set, self.get_word_vector(w))
                similarity_score[w] += self.compute_cosine_similarity(v_vect, w_vect, words_set)
        return similarity_score