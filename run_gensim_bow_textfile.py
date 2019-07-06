import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from smart_open import smart_open
import nltk
nltk.download('stopwords') # run once
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

"""
smart_open is pretty great as it opens files line by line, as opposed to simply open
nltk is the natural language toolkit that helps us work with human language data
"""


class BoWCorpus(object):
    def __init__(self, path, dictionary):
        self.filepath = path
        self.dictionary = dictionary

    def __iter__(self):
        global mydict # OPTIONAL, only if updating the source dictionary
        for line in smart_open(self.filepath, encoding='latin'):
            # tokenize
            tokenised_list = simple_preprocess(line, deacc=True)

            #create bag of words
            bow = self.dictionary.doc2bow(tokenised_list, allow_update=True)

            #update the source dictionary (OPTIONAL)
            mydict.merge_with(self.dictionary)

            #lazy return the BoW
            yield bow

#create the dictionary
mydict = corpora.Dictionary()

#create the corpus
bow_corpus = BoWCorpus('sample.txt', dictionary=mydict) #memory friendly

#print the token_id and count for each line
for line in bow_corpus:
    print(line)

# Save the Dict and Corpus
mydict.save('mydict.dict')  # save dict to disk
corpora.MmCorpus.serialize('bow_corpus.mm', bow_corpus)  # save corpus to disk

# Load them back
loaded_dict = corpora.Dictionary.load('mydict.dict')

corpus = corpora.MmCorpus('bow_corpus.mm')
for line in corpus:
    print(line)