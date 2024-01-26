import pandas as pd
import re
from nltk.tokenize import WordPunctTokenizer
from nltk.lm.preprocessing import pad_both_ends
from nltk import ngrams
from itertools import chain


class Processing:

    def n_grams(self, sentence, vocabulary, pad=True, n=2):
        sentence = self.__replace_unk_words(vocabulary, sentence)
        if pad:
            return list(ngrams(pad_both_ends(self.__replace_unk_words(vocabulary, sentence), n=n), n=n))
        else:
            return list(ngrams(self.__replace_unk_words(vocabulary, sentence), n=n))


    def __replace_unk_words(self, vocabulary, sentence):
        for i in range(len(sentence)):
            sentence[i] = vocabulary.lookup(sentence[i])
        return sentence
    

class Preprocessing:

    def preprocess_corpus(self, corpus):    # Assume df already loaded
        if type(corpus) != pd.DataFrame:
            raise TypeError("corpus argument is not a DataFrame\n")
        corpus['Body'] = corpus['Body'].apply(lambda x: self.__remove_html_tags(x))
        corpus['Body_tokenized'] = corpus['Body'].apply(lambda x: WordPunctTokenizer().tokenize(x))
        return corpus
    
    
    def __remove_html_tags(self, text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
