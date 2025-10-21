import nltk
import re
import streamlit as st
import pickle

nltk.download('punct')
nltk.download('stopwords')

svc_model = pickle.load(open('svc.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('le.pkl', 'rb'))

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', '  ', cleanText)
    cleanText = re.sub('@\S+', ' ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def main():
    st.title('Resume Screening App')
    upload_file = st.file_uploader("Upload Resume", type=['txt', 'pdf'])
    
    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_txt = resume_bytes.decode('utf-8')
        except:
            resume_txt = resume_bytes.decode('latin-1')
        
        cleaned_resume = cleanResume(resume_txt)
        cleaned_resume = tfidf.transform([cleaned_resume])
        predicted_lable = svc_model.predict(cleaned_resume)[0]
        st.write(le.inverse_transform([predicted_lable])[0])
    
if __name__ == "__main__":
    main()