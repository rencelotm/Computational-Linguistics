import pandas as pd
from preprocess import *
from vocabulary import *
from Bayes import test_correctness
from nltk.probability import ConditionalFreqDist
from numpy import log, unique


class SmoothedBigramsBag:

    def __init__(self, y, n, corpus, vocabulary):
        self.bag = dict()
        self.__n_history = n - 1
        self.__classes = unique(y)
        self.prior = list()
        self.__compute_prior(y)
        self.__init_bag()
        self.__construct_bag(corpus, y, vocabulary)

    def get_classes(self):
        return self.__classes
    
    def get_history_order(self):
        return self.__n_history
    
    def get_count_context(self, cla, ctx):
        return sum(self.bag[cla][ctx].values())

    def __init_bag(self):
        for cla in self.__classes:
            self.bag[cla] = ConditionalFreqDist()
    
    def __construct_bag(self, corpus, y, vocabulary):
        for i in range(len(corpus)):
            cla = y[i]
            sentence = Processing().n_grams(corpus[i], vocabulary, pad=True, n=self.__n_history + 1)
            for grams in sentence:
                ctx = grams[:self.__n_history]
                word = grams[-1]
                self.bag[cla][ctx][word] += 1
    
    def __compute_prior(self, y):
        for cla in self.__classes:
            self.prior.append(sum([1 if i == cla else 0 for i in y]) /len(y))


class NGramsModels:

    def __init__(self, corpus, y, vocabulary, bag_class, n):
        self.bag = bag_class(y, n, corpus, vocabulary)
        self.vocabulary = vocabulary
        self.__vocab_length = len(vocabulary)
    
    def predict_sentence_class(self, sentence, epsilon):
        sentence = Processing().n_grams(sentence, self.vocabulary, pad=True, n=self.bag.get_history_order()+1)
        class_probabilities = [0] * len(self.bag.get_classes())
        for i in range(len(class_probabilities)):
            for gram in sentence:
                cla = self.bag.get_classes()[i]
                ctx = gram[:self.bag.get_history_order()]
                word = gram[-1]
                class_probabilities[i] += log((self.bag.bag[cla][ctx][word] + epsilon) / (self.bag.get_count_context(cla, ctx) + self.__vocab_length * epsilon))
            class_probabilities += log(self.bag.prior[i])
        idx = list(class_probabilities).index(max(class_probabilities))
        return self.bag.get_classes()[idx]
    
    def predict(self, test_corpus, epsilon):
        res = list()
        for sentence in test_corpus:
            res.append(self.predict_sentence_class(sentence, epsilon))
        return res


if __name__ == "__main__":
    df_corpus = Preprocessing().preprocess_corpus(pd.read_csv("corpora/train.csv"))
    df_test = Preprocessing().preprocess_corpus(pd.read_csv("corpora/test.csv"))
    vocabulary = VocabularyConstructor().construct_vocabulary(df_corpus['Body_tokenized'], 3)
    bigrams_model = NGramsModels(df_corpus["Body_tokenized"], df_corpus['Y'], vocabulary, SmoothedBigramsBag, 2)

    predicted_test = bigrams_model.predict(df_test['Body_tokenized'], 0.1)
    print(test_correctness(predicted_test, df_test['Y']))

    predicted_test = bigrams_model.predict(df_test['Body_tokenized'], 0.01)
    print(test_correctness(predicted_test, df_test['Y']))

    predicted_test = bigrams_model.predict(df_test['Body_tokenized'], 0.001)
    print(test_correctness(predicted_test, df_test['Y']))

    predicted_test = bigrams_model.predict(df_test['Body_tokenized'], 0.0001)
    print(test_correctness(predicted_test, df_test['Y']))
