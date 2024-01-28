from nltk.lm import Vocabulary


class VocabularyConstructor:

    def construct_vocabulary(self, words, cutoff):
        return Vocabulary(words, unk_cutoff=cutoff)