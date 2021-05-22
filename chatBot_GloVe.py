import json
import numpy as np
import tensorflow as tf
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder

training_sentences, training_labels = [], []
labels, responses = [], []
import pickle

#print(tf.__version__, '\t\t', tf.__file__,'\n')

with open('intents.json') as f:
    data = json.load(f)

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])

    responses.append(intent['responses'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

#print(dict(zip(labels,responses)))

enc = LabelEncoder()
enc.fit(training_labels)  ## 'Y' | Dependent Variable
training_labels = enc.transform(training_labels)

vocab_size = 8000
max_len = 24
trunc_type = "post"
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)

padded = pad_sequences(sequences, truncating = trunc_type, maxlen = max_len)
classes = len(labels)

'''
print(type(padded))
print('\n\n')
print(type(sequences),'\n')
print(padded[5], '\n\n', sequences[5])
print(word_index['whats'], word_index['up'])
print(training_sentences[5])
'''

embeddings_index = {}
with open('../glove.6B/glove.6B.200d.txt', encoding= 'utf-8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors: " % len(embeddings_index))
num_tokens = len(word_index)+2
print('num_tokens- ', num_tokens)
embedding_dim = 200
hits, misses = 0, 0
EPOCHS = 240

embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        hits+=1
    else:
        misses+=1

print("Converted %d words (%d missed)" % (hits, misses))

from tensorflow.keras.layers import Embedding

model = tf.keras.models.Sequential()
model.add(Embedding(num_tokens, 200, weights=[embedding_matrix],
                    input_length = 24, trainable= False))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(75)))
model.add(tf.keras.layers.Dense(32, activation = 'relu'))
model.add(tf.keras.layers.Dense(classes, activation = 'softmax'))
model.summary()


model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(padded, np.array(training_labels), epochs=EPOCHS)
tf.keras.models.save_model(model, 'GloVe200d_LSTM_chatBot_v1')


with open('chatBotTokenizer_v1.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
