import streamlit as st
import pickle
import re

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

st.title("Spam Message Detector")

msg = st.text_area("Enter your message")

if st.button("Predict"):

    cleaned = clean_text(msg)

    vec = vectorizer.transform([cleaned])

    prediction = model.predict(vec)[0]

    if prediction == 1:
        st.error("Spam Message")
    else:
        st.success("Ham Message")