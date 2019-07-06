import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from smart_open import smart_open
import os

class ReadTxtFiles(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding='latin'):
                yield simple_preprocess(line)

path_to_text_directory = "lsa_sports_food_docs"

dictionary = corpora.Dictionary(ReadTxtFiles(path_to_text_directory))

# Token to Id map
print(dictionary.token2id)

