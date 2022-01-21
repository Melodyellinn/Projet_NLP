#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:19:51 2022

@author: Simplon
"""

import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer



df = pd.read_csv("C:/Users/Simplon/Documents/meloyellinn/NLP_projet/NLP/data/spam_newcol.csv")

tfidf = pickle.load(open('vectorizer.pkl','rb'))
corpus = pickle.load(open('corpus.pkl', 'rb'))

def transform_text(text):
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    X = tfidf.fit_transform(corpus)
    y = df["CATEGORY"]
    return corpus


model = pickle.load(open('model.pkl','rb'))


st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform(transformed_sms)
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

