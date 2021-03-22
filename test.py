# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:52:29 2021

@author: Agnij
"""
import pickle
from tensorflow import keras
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import json

#from sklearn.preprocessing import LabelEncoder

# loading
with open('tokenizer_v1.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
with open('intents.json') as f:
    data = json.load(f)
    
#model = tf.keras.models.load_model('simpleChatBot_v1')
model = keras.models.load_model('simpleChatBot_v1')
labels = ['greeting', 'goodBye', 'buy', 'timePass', 'customerCare']
max_len = 20
trunc_type = 'post'

'''
def trial():
    print("Commence Speech, 'quit' to Exit")
    while True:
        string = input("Enter: ")
        if string == 'quit':
            break
        result = model.predict(pad_sequences(tokenizer.texts_to_sequences([string]),
                                             truncating=trunc_type, maxlen = max_len))
        
        #print(type(result[0]))
        pos = list(result[0]).index(max(result[0]).item())
        category = sorted(labels)[pos]
        
        print(max(result[0]).item())

        if max(result[0]).item() > 0.6:
            for i in data['intents']:
                if i['tag'] == category:
                    print(category,':  ',np.random.choice(i['responses']))
        else:
            print('Unable to comprehend the vastness of this life, please try in another dimension')
                
                
trial()
'''


def getpred(sentence):
    result = model.predict(pad_sequences(tokenizer.texts_to_sequences([sentence]),
                                             truncating=trunc_type, maxlen = max_len))
    #print(type(result[0]))
    pos = list(result[0]).index(max(result[0]).item())
    category = sorted(labels)[pos]
    
    #print(max(result[0]).item())

    if max(result[0]).item() > 0.6:
        for i in data['intents']:
            if i['tag'] == category:
                return category+':  '+np.random.choice(i['responses'])
    else:
        return 'Unable to comprehend the vastness of this life, please try in another dimension'



import gradio as gr

iface = gr.Interface(fn=getpred, inputs="text", outputs="text")
iface.launch(share = True)
