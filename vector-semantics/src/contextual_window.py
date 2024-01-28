from nltk.probability import ConditionalFreqDist, FreqDist
from math import sqrt, log10

class ContextualWindowConstructor:

    def __init__(self, size):
        self.__size = size // 2

    def get_size(self):
        return self.__size
    
    def get_contextual_window(self, word_idx, corpus):
        if word_idx < 0 or word_idx >= len(corpus):
            raise IndexError("Target word index is out of range")
        start = max(0, word_idx - self.__size)
        end = min(len(corpus), word_idx + self.__size + 1)
        return ContextualWindow(corpus[start:word_idx], corpus[word_idx], corpus[word_idx + 1:end])


class ContextualWindow:

    def __init__(self, left_context, word, right_context):
        self.left_context = left_context
        self.word = word
        self.right_context = right_context
    
    def get_contextual_window_without_word(self):
        return self.left_context + self.right_context
    
    def get_contextual_window_with_word(self):
        return self.left_context + list(self.word) + self.right_context
    
    def __str__(self):
        return ("{}".format(",".join(self.get_contextual_window_without_word())))
