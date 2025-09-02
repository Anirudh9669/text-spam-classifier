import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy.sparse import hstack
import re

# --- NLTK Downloads (required for first run) ---
# This ensures that the necessary data is downloaded in the Streamlit Cloud environment.
# It only runs once when the app is first deployed or when a new version is pushed.
with st.spinner('Downloading necessary data...'):
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


@st.cache_resource
def load_models():
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return tfidf, model, scaler
    except FileNotFoundError:
        st.error("Model, vectorizer, or scaler files not found. Please run the Jupyter notebook first.")
        st.stop()


tfidf, model, scaler = load_models()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)

        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Create and scale the new numerical feature
        num_characters = len(re.sub(r'[^a-zA-Z0-9]', '', input_sms))
        numerical_features = [[num_characters]]
        numerical_features_scaled = scaler.transform(numerical_features)

        # 4. Combine vectorized text with the new feature
        combined_input = hstack([vector_input, numerical_features_scaled])

        # 5. Predict
        result = model.predict(combined_input.toarray())[0]

        # 6. Display
        st.divider()
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
