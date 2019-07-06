import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from smart_open import smart_open
import os

#create gensim dictionary from a single text file
dictionary = corpora.Dictionary(simple_preprocess(line,deacc=True) for line in open('sample.txt',encoding='utf-8'))
#deacc is deaccented by the way


#see token to id
print(dictionary.token2id)


