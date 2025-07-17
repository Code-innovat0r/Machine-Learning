''' To Run this app type 'streamlit run app.py' this in the trminal '''


import pickle
import streamlit as st
import string
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def text_transform(text: str):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
         if i.isalnum(): # filter a-z + A-Z + 0-9  , isalpha - filter  A-Z or a-z
             y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # converting the word into its root form
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classification")

input_sms = st.text_input("Enter a message : ")

if st.button('Predict'):
    transform_text = text_transform(input_sms)

    vectorized_string = tfidf.transform([transform_text])

    prediction = model.predict(vectorized_string)

    if prediction == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


