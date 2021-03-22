import json
import numpy as np
import tensorflow as tf
#import keras
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
#from tensorflow.keras.models import Sequential, load_model

from sklearn.preprocessing import LabelEncoder

training_sentences, training_labels = [], []
labels, responses = [], []
import pickle

with open('intents.json') as f:
    data = json.load(f)

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

enc = LabelEncoder()
enc.fit(training_labels)
training_labels = enc.transform(training_labels)

##print(training_labels)

vocab_size = 10000
embedding_dim = 16
max_len = 20
trunc_type = 'post'
#padding = 'pre'
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)

padded = pad_sequences(sequences, truncating=trunc_type, maxlen=max_len)
classes = len(labels)

'''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_len))

model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))

model.add(tf.keras.layers.Dense(classes, activation='softmax'))

model.summary()
training_labels_final = np.array(training_labels)

EPOCHS = 300
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(padded, training_labels_final, epochs=EPOCHS)
'''

def trial():
    print("Commence Speech, 'quit' to Exit")
    while True:
        string = input("Enter: ")
        if string == 'quit':
            break
        result = model.predict(pad_sequences(tokenizer.texts_to_sequences([string]),
                                             truncating=trunc_type, maxlen = max_len))
        
        category = enc.inverse_transform([np.argmax(result)])
        
        print(max(result[0]).item())

        if max(result[0]).item() > 0.6:
            for i in data['intents']:
                if i['tag'] == category:
                    print(category,':  ',np.random.choice(i['responses']))
        else:
            print('Unable to comprehend the vastness of this life, please try in another dimension')
                
                
trial()
#tf.keras.models.save_model(model, 'simpleChatBot_v1')
model = tf.keras.models.load_model('simpleChatBot_v1')


with open('tokenizer_v1.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)