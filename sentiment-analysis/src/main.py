from preprocess import *
from vocabulary import *
import pandas as pd


if __name__ == "__main__":
    df_corpus = Preprocessing().preprocess_corpus(pd.read_csv("corpora/train.csv"))
    train_corpus = df_corpus['Body_tokenized']
    train_vocabulary = VocabularyConstructor().construct_vocabulary(train_corpus, 3)
    
    print("Last n tokens of the vocabulary in alphabetical order: \n")
    print(VocabularyMethods().get_n_last_tokens(10, train_vocabulary))

    print("Out-Of-Vocabulary rate before pad sentences with <s> and </s>: \n")
    print(VocabularyMethods().oov_rate(train_vocabulary, train_corpus))

    print("Padded sentence at index 8198: \n")
    print(Processing().n_grams(train_corpus[8198], train_vocabulary))