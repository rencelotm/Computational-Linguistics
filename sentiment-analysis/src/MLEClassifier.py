from preprocess import *
from vocabulary import *
from nltk.probability import ConditionalFreqDist
from numpy import log2


class Estimator:

    def __init__(self, n, corpus, cutoff=3):
        self.__n_history = n - 1
        self.vocabulary = VocabularyConstructor().construct_vocabulary(corpus, cutoff=cutoff)
        self.frequencies = self.__build_model(corpus)
    
    def get_history(self):
        return self.__n_history
    
    def __build_model(self, corpus):
        frequencies = ConditionalFreqDist()
        n = self.__n_history + 1
        for sentence in corpus:
            sentence = Processing().n_grams(sentence, self.vocabulary, n=n)
            for bi in sentence:
                ctx = bi[0:self.__n_history]
                word = bi[-1]
                frequencies[tuple(ctx)][word] += 1
        return frequencies
    

class MaximumLikelihoodEstimator(Estimator):

    def __init__(self, n, corpus, cutoff=3):
        super().__init__(n, corpus, cutoff)
    
    def get_n_most_likely_word(self, n, ctx):
        if type(ctx) != tuple:
            raise TypeError("You must have to pass ctx as a tuple even if context is a single word!\n")
        ret = dict()
        tmp = self.frequencies[ctx]
        most_likely = tmp.most_common(n=n)
        for word, _ in most_likely:
            ret[word] = round(tmp.freq(word), 2)
        return ret
    
    def get_probability(self, test_ngrams):
        if len(test_ngrams) > self.get_history() + 1:
            raise IndexError("The ngrams used is too long!\n")
        ctx = test_ngrams[:self.get_history()]
        word = test_ngrams[-1]
        c_ctx = sum(self.frequencies[ctx].values())
        return (self.frequencies[ctx][word]) / (c_ctx)
    
    def compute_perplexity(self, test_set):
        M = 0
        score = 0
        for sentence in test_set:
            s_ngram = Processing().n_grams(sentence, self.vocabulary)
            for gram in s_ngram:
                M += 1
                score += log2(self.get_probability(gram))
        LL = score / M
        return round(2 ** (-LL), 3)



class AddOneLaplaceSmoothing(MaximumLikelihoodEstimator):

    def __init__(self, n, corpus, cutoff=3):
        super().__init__(n, corpus, cutoff)
        self.words = self.__preprocess_words()
    
    def __preprocess_words(self):
        words = set()
        for word in self.vocabulary.counts:
            words.add(self.vocabulary.lookup(word))
        return words

    def get_n_most_likely_word(self, n, ctx):
        if type(ctx) != tuple:
            raise TypeError("You must have to pass ctx as a tuple even if context is a single word!\n")
        tmp = dict()
        ret = dict()
        c_ctx = sum(self.frequencies[ctx].values())
        for word in self.words:
            tmp[word] = (self.frequencies[ctx][word] + 1) / (c_ctx + len(self.vocabulary))
        tmp = sorted(tmp.items(), key=lambda x:x[1], reverse=True)[:n]
        for elem in tmp:
            ret[elem[0]] = round(elem[1], 2)
        return ret
    
    def get_probability(self, test_ngrams):
        if len(test_ngrams) > self.get_history() + 1:
            raise IndexError("The ngrams used is too long!\n")
        ctx = test_ngrams[:self.get_history()]
        word = test_ngrams[-1]
        c_ctx = sum(self.frequencies[ctx].values())
        return (self.frequencies[ctx][word] + 1) / (c_ctx + len(self.vocabulary))

    def compute_perplexity(self, test_set):
        M = 0
        score = 0
        for sentence in test_set:
            s_ngram = Processing().n_grams(sentence, self.vocabulary)
            for gram in s_ngram:
                M += 1
                score += log2(self.get_probability(gram))
        LL = score / M
        return round(2 ** (-LL), 3)


if __name__ == "__main__":
    df_corpus = Preprocessing().preprocess_corpus(pd.read_csv("corpora/train.csv"))
    df_test = Preprocessing().preprocess_corpus(pd.read_csv("corpora/test.csv"))
    train_corpus = df_corpus['Body_tokenized']
    test_corpus = df_test['Body_tokenized']
    mle = MaximumLikelihoodEstimator(2, train_corpus, 3)
    print(mle.get_n_most_likely_word(3, ('<s>',)))
    print(mle.compute_perplexity(test_corpus))
    laplace = AddOneLaplaceSmoothing(2, train_corpus, 3)
    print(laplace.get_n_most_likely_word(3, ('<s>',)))
    print(laplace.compute_perplexity(test_corpus))