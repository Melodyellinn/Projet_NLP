# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:49:44 2022

@author: Simplon
"""

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import tensorflow.keras.preprocessing 
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    def get_sequences(texts, tokenizer, train=True, max_seq_length=None):
        sequences = tokenizer.texts_to_sequences(texts)
        if train == True:
            max_seq_length = np.max(list(map(lambda x: len(x), sequences)))
            sequences = tensorflow.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_seq_length, padding='post')

        return sequences

    def preprocess_test(df):
        df = df.copy()
    
        # Drop FILE_NAME column
        df = df.drop('FILE_NAME', axis=1)
    
        # Split df into X and y
        y = df['CATEGORY']
        X = df['MESSAGE']
   
        # Create tokenizer
        tokenizer = pickle.load(open("tokenizer.pickle","rb"))
        X_train_len = 14804
    
        # Convert texts to sequences
        X_test = get_sequences(X, tokenizer, train=False, max_seq_length=X_train_len)
        return X_test, y

        cv = CountVectorizer()
        model = load_model('model.h5')
        
        if request.method == 'POST':
            #message = request.form['MESSAGE']
            vect = cv.transform(X_test).toarray()
            #data = [message]
            my_prediction = model.predict(vect)
        return render_template('index.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run()