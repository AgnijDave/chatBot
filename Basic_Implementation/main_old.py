import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tensorflow
#import tflearn
import random

stemmer = LancasterStemmer()

with open('intents.json') as file:
    data = json.load(file)

#print(data)
words,labels,docs_x,docs_y = [],[],[],[]

for intent in data['intents']:
    for pattern in intent["patterns"]:
        wds = nltk.word_tokenize(pattern)
        words.extend(wds)

        docs_x.append(wds)
        docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent['tag'])

#print(words)
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
#print(words)
words = sorted(list(set(words)))
labels = sorted(labels)

print(words,labels)
print(docs_x,docs_y)

training,output = [],[]
out_empty = [0 for _ in range(len(labels))]

#docs_x = ['ola']
for x,doc in enumerate(docs_x):
    bag = []

    wds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    #print(output_row)

    training.append(bag)
    output.append(output_row)

print(training)
print(output)

training = numpy.array(training)
output = numpy.array(output)

