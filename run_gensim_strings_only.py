import gensim
from gensim import corpora
from pprint import pprint

# creating dictionary from a list of sentences
documents = ["The Saudis are preparing a report that will acknowledge that",
             "Saudi journalist Jamal Khashoggi's death was the result of an",
             "interrogation that went wrong, one that was intended to lead",
             "to his abduction from Turkey, according to two sources."]

documents_2 = ["One source says the report will likely conclude that",
                "the operation was carried out without clearance and",
                "transparency and that those involved will be held",
                "responsible. One of the sources acknowledged that the",
                "report is still being prepared and cautioned that",
                "things could change."]

# tokenize the sentences or documents into words
texts = [[text for text in doc.split()] for doc in documents]


# create the dictionary
dictionary = corpora.Dictionary(texts)


# get information about the dictionary

print(dictionary)

# ...dictionary has 33 unique words

pprint(dictionary.token2id)

#adding more items to the dictionary
texts2 = [[text for text in doc.split()] for doc in documents_2]
dictionary.add_documents(texts2)

#now the dictionary will have 60 unique words
print(dictionary)
pprint(dictionary.token2id)


