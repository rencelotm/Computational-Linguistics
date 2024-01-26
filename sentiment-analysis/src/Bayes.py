import pandas as pd
from preprocess import *
from vocabulary import *
from nltk.probability import ConditionalFreqDist, FreqDist
from numpy import log, unique


class Bag:

    def __init__(self, y):
        self.bag = ConditionalFreqDist()
        self.__classes = unique(y)
        self.prior = list()
        self.__compute_prior(y)
    
    def get_classes(self):
        return self.__classes
    
    def get_total_word_in_class(self, cla):
        if cla not in self.get_classes():
            raise ValueError("Value for parameter 'cla is not correct!\n")
        return sum(self.bag[cla].values())

    def __compute_prior(self, y):
        for cla in self.__classes:
            self.prior.append(sum([1 if i == cla else 0 for i in y]) /len(y))


class NaiveBag(Bag):

    def __init__(self, corpus, y, vocabulary):
        super().__init__(y)
        self.__construct_bag(corpus, y, vocabulary)
    
    def __construct_bag(self, corpus, y, vocabulary):
        for i in range(len(corpus)):
            cla = y[i]
            sentence = corpus[i]
            for word in sentence:
                if vocabulary.lookup(word) != '<UNK>':
                    self.bag[cla][word] += 1


class BinaryBag(Bag):

    def __init__(self, corpus, y, vocabulary):
        super().__init__(y)
        self.__construct_bag(corpus, y, vocabulary)
    
    def __construct_bag(self, corpus, y, vocabulary):
        for i in range(len(corpus)):
            cla = y[i]
            sentence = unique(corpus[i])
            for word in sentence:
                if vocabulary.lookup(word) != '<UNK>':
                    self.bag[cla][word] += 1


class NegativeBag(Bag):

    def __init__(self, corpus, y, vocabulary):
        super().__init__(y)
        self.__construct_bag(corpus, y, vocabulary)
    
    def __construct_bag(self, corpus, y, vocabulary):
        for i in range(len(corpus)):
            cla = y[i]
            sentence = self.__process_negative_sentence(corpus[i], vocabulary)
            for word in sentence:
                if word != '<UNK>':
                    self.bag[cla][word] += 1

    def __process_negative_sentence(self, sentence, vocabulary, punctuation=['.', ',', ':', '?', '!'], negative=['not', 'no', 'never']):
        neg = False
        for i in range(len(sentence)):
            if sentence[i] in negative and not neg:
                neg = True
            elif sentence[i] in punctuation:
                neg = False
            elif neg:
                sentence[i] = sentence[i] + '_NOT'
            else:
                sentence[i] = vocabulary.lookup(sentence[i])
        return sentence


class NaiveBayes:

    def __init__(self, corpus, y, vocabulary, bag_class):
        self.bag = bag_class(corpus, y, vocabulary)
        self.__vocab_length = len(vocabulary)
    
    def predict_sentence_class(self, sentence):
        class_probabilities = [0] * len(self.bag.get_classes())
        for i in range(len(class_probabilities)):
            for word in sentence:
                cla = self.bag.get_classes()[i]
                class_probabilities[i] += log((self.bag.bag[cla][word] + 1) / (self.bag.get_total_word_in_class(cla) + self.__vocab_length))
            class_probabilities += log(self.bag.prior[i])
        idx = list(class_probabilities).index(max(class_probabilities))
        return self.bag.get_classes()[idx]
    
    def predict(self, test_corpus):
        res = list()
        for sentence in test_corpus:
            res.append(self.predict_sentence_class(sentence))
        return res


def test_correctness(predicted, labels):
    return round(sum([1 if predicted[i] == labels[i] else 0 for i in range(len(labels))]) / len(labels) * 100, 3)


if __name__ == '__main__':
    df_corpus = Preprocessing().preprocess_corpus(pd.read_csv("corpora/train.csv"))
    df_test = Preprocessing().preprocess_corpus(pd.read_csv("corpora/test.csv"))
    vocabulary = VocabularyConstructor().construct_vocabulary(df_corpus['Body_tokenized'], 3)
    naive_bayes = NaiveBayes(df_corpus['Body_tokenized'], df_corpus['Y'], vocabulary, NaiveBag)

    predicted_test = naive_bayes.predict(df_test['Body_tokenized'])
    print(test_correctness(predicted_test, df_test['Y']))

    binary_bayes = NaiveBayes(df_corpus['Body_tokenized'], df_corpus['Y'], vocabulary, BinaryBag)
    predicted_test = binary_bayes.predict(df_test['Body_tokenized'])
    print(test_correctness(predicted_test, df_test['Y']))

    negative_bayes = NaiveBayes(df_corpus['Body_tokenized'], df_corpus['Y'], vocabulary, NegativeBag)
    predicted_test = negative_bayes.predict(df_test['Body_tokenized'])
    print(test_correctness(predicted_test, df_test['Y']))
