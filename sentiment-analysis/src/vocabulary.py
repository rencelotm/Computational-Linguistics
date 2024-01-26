from nltk.probability import FreqDist, ConditionalFreqDist, MLEProbDist, ConditionalProbDist
from nltk.lm import Vocabulary
from itertools import chain

class VocabularyConstructor:

    def construct_vocabulary(self, corpus, cutoff):
        return Vocabulary(self.__compute_words_frequency(corpus), unk_cutoff=cutoff)
    
    def __compute_words_frequency(self, corpus):
        frequencies = FreqDist()
        for word in chain.from_iterable(corpus):
            frequencies[word] += 1
        return frequencies
    

class VocabularyMethods:

    def get_n_last_tokens(self, n, vocabulary):
        res = list()
        for word in vocabulary.counts.keys():
            if vocabulary.lookup(word) != '<UNK>':
                res.append((word, vocabulary.counts[word]))
        return sorted(res, key=lambda x: (x[0]))[len(vocabulary) - n - 1:]
    
    def oov_rate(self, vocabulary):
        n_words = 0
        n_unks = 0
        for word in vocabulary.counts.keys():
            if vocabulary.lookup(word) == '<UNK>':
                n_unks += vocabulary.counts[word]
            n_words += vocabulary.counts[word]
        return round(n_unks / n_words * 100, 3)
