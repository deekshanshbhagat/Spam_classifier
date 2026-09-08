import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  y = []
  # appending alpha_numeric(removing special characters)
  for i in text:
    if i.isalnum():
      y.append(i)
  text = y[:]  # cloning
  y.clear()
  # appending non-stopwords
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Spam Classifier")

input_sms = st.text_area("Enter your message")
if st.button("Predict"):
    # 1 preprocess
    transformed_sms = transform_text(input_sms)
    # 2 vectorization
    vector_input = tfidf.transform([transformed_sms])
    # 3 predict
    prediction = model.predict(vector_input)[0]
    probabilities = model.predict_proba(vector_input)[0]
    # 4 Display
    if prediction == 1:
        st.header("🚨 Spam")
        st.markdown(f"**🧮 Probability of Spam:** `{probabilities[1] * 100:.2f}%`")
    else:
        st.header("✅ Not Spam")
        st.markdown(f"**🛡️ Probability of Not Spam:** `{probabilities[0] * 100:.2f}%`")
