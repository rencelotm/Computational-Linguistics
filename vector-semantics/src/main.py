from preprocess import *
from vocabulary import *
from contextual_window import *
from vector import *


if __name__ == "__main__":
    corpus = Preprocessing().preprocess_corpus('corpus.txt')
    # Test Efficiency of FreqDist() solution for frequencies of corpus' words
    corpus_information_giver = TextInformation(corpus)
    vocabulary = VocabularyConstructor().construct_vocabulary(corpus, 10)
    corpus = Preprocessing().transform_text(corpus, vocabulary)
    matrix = TFIDF(corpus, 5)
    #print(compute_cosine_similarity(matrix.get_word_vector('languages'), matrix.get_word_vector('linguists'), set(corpus)))
    print(matrix.get_k_closest_word('car', set(corpus), 10))
    
