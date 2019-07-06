import gensim
from gensim import corpora
from gensim.utils import simple_preprocess

#corpus has word_id and frequency inside it - we will start with a list with 2 sentences

my_doc = ["Who let the dogs out?",
           "Who? Who? Who? Who?"]

#tokenising

tokenised_list = [simple_preprocess(doc) for doc in my_doc]

#time to create the corpus

mydict = corpora.Dictionary()
mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in tokenised_list]
print(mycorpus)

#count all the words in the corpus

word_counts_corpus = [[(mydict[id], count) for id, count in line] for line in mycorpus]
print(word_counts_corpus)