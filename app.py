import nltk
import re
import streamlit as st
import pickle

nltk.download('punct')
nltk.download('stopwords')

svc_model = pickle.load(open('svc.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

st.title('Resume Screening App')