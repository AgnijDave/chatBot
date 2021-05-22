from tensorflow import keras
from keras_preprocessing.sequence import pad_sequences

import pickle
import numpy as np

with open('chatBotTokenizer_v1.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

import json
with open('intents.json') as f:
    data = json.load(f)

model = keras.models.load_model('GloVe200d_LSTM_chatBot_v1')
classes = ['greeting', 'goodBye', 'buy', 'timePass', 'customerCare']

max_len = 24
trunc_type = 'post'

def getpred(sentence):
    result = model.predict(pad_sequences(tokenizer.texts_to_sequences([sentence.lower()]),
                            truncating=trunc_type, maxlen = max_len))

    #print(result)
    pos = list(result[0]).index(max(result[0]).item())
    category = sorted(classes)[pos]
    #print([pos] ,'\t', category)

    if max(result[0]).item() > 0.7:
        for d in data['intents']:
            if d['tag'] == category:
                return category+' |%f| ' % max(result[0]).item() +': '+np.random.choice(d['responses'])

    else:
        return 'Please Try Again || %f' % max(result[0]).item()
    
    
getpred('zwemlesgever tijdens mijn vrije tijd')
getpred('HI can i speak to customer exec')