import re
from nltk.corpus.reader import PlaintextCorpusReader
from nltk.probability import FreqDist


PUNCTUATIONS = [".", ",", ":", ";", "!", "?", "-", "\'", "\"", "(", ")", "[", "]"]


class Preprocessing:

    def preprocess_corpus(self, filename):
        corpus = PlaintextCorpusReader(root="corpora", fileids=[filename]).words()
        # Construct Regex
        punct = '|'.join(re.escape(p) for p in PUNCTUATIONS)
        punct_pattern = re.compile(r'[{}]'.format(punct))
        # Preprocessing Tasks
        words = list()
        for word in corpus:
            word = self.__delete_punctuation(self.__lowerize_tokens(word), punct_pattern)
            if word != '':    #?? Really Useful ?? -> Branching not good for processor pipelining !!
                words.append(word)
        return words
    
    def transform_text(self, corpus, vocabulary):
        for i in range(len(corpus)):
            corpus[i] = vocabulary.lookup(corpus[i])
        return corpus

    def __lowerize_tokens(self, word):
        return word.lower()

    def __delete_punctuation(self, word, punctuations):
        return punctuations.sub('', word)


class TextInformation:

    def __init__(self, corpus):
        self.__frequencies = FreqDist(corpus)

    def n_common_words(self, n):    # Corpus have to be parsed first !
        return sorted(self.__frequencies.most_common(n), key=lambda tup: (-tup[1], tup[0]))
