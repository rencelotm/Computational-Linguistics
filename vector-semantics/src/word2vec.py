import gensim
import gensim.downloader
import re
import numpy as np
import nltk
from nltk.corpus.reader import PlaintextCorpusReader
from vocabulary import *



PUNCTUATIONS = [".", ",", ":", ";", "!", "?", "-", "\'", "\"", "(", ")", "[", "]"]


class Word2VecPreprocessing:

    def preprocess_corpus(self, filename):
        corpus = PlaintextCorpusReader(root="corpora", fileids=[filename]).sents()
        # Construct Regex
        punct = '|'.join(re.escape(p) for p in PUNCTUATIONS)
        punct_pattern = re.compile(r'[{}]'.format(punct))
        # Preprocessing Tasks
        sentences = list()
        for sentence in corpus:
            tmp = list()
            for word in sentence:
                word = self.__delete_punctuation(self.__lowerize_tokens(word), punct_pattern)
                if word != '':
                    tmp.append(word)
            sentences.append(tmp)
        return sentences
    
    def transform_text(self, corpus, vocabulary):
        for i in range(len(corpus)):
            for j in range(len(corpus[i])):
                corpus[i][j] = vocabulary.lookup(corpus[i][j])
        return corpus
    
    def __lowerize_tokens(self, word):
        return word.lower()
    
    def __delete_punctuation(self, word, punctuations):
        return punctuations.sub('', word)


if __name__ == "__main__":
    nltk.download('punkt')
    corpus = Word2VecPreprocessing().preprocess_corpus('corpus.txt')
    print("Text transformation begins!")
    #model = gensim.models.Word2Vec(corpus, vector_size=100, window=5, sg=1, negative=10, epochs=50, min_count=10, workers=1)
    #print(model.wv.most_similar('car', topn=5))
    #print(model.wv.most_similar('feature', topn=5))
    #print(model.wv.most_similar('computer', topn=5))

    print("Most Similar to computer with pre-trained model from model")
    second_model = gensim.downloader.load('word2vec-google-news-300')
    print(second_model.most_similar('computer', topn=5))
