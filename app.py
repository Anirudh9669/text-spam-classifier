import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from scipy.sparse import hstack
import re

# Set page configuration for a wider, cleaner layout
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📨",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# Load resources with caching for performance
@st.cache_resource
def load_models():
    """Loads and caches the models and scaler from pickle files."""
    try:
        tfidf_model = pickle.load(open('vectorizer.pkl', 'rb'))
        spam_model = pickle.load(open('model.pkl', 'rb'))
        scaler_model = pickle.load(open('scaler.pkl', 'rb'))
        return tfidf_model, spam_model, scaler_model
    except FileNotFoundError:
        st.error(
            "Model files not found. Please ensure 'vectorizer.pkl', 'model.pkl', and 'scaler.pkl' are in the same directory.")
        return None, None, None


@st.cache_resource
def download_nltk_data():
    """Downloads necessary NLTK data if not present."""
    with st.spinner('Downloading necessary language data...'):
        try:
            nltk.data.find('corpora/stopwords')
        except:
            nltk.download('stopwords')
        try:
            nltk.data.find('tokenizers/punkt')
        except:
            nltk.download('punkt')


# Preprocess the text
def transform_text(text):
    """
    Transforms raw text by lowercasing, tokenizing, removing stopwords,
    punctuation, and stemming the words.
    """
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
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Main app layout
st.title("Email/SMS Spam Classifier 📨")
st.markdown("Enter a message below to predict if it's spam or not.")
st.divider()

# Load models and NLTK data
download_nltk_data()
tfidf, model, scaler = load_models()

if tfidf and model and scaler:
    input_sms = st.text_area("Enter the message here:", height=150)

    if st.button('Predict'):
        if input_sms:
            # 1. Preprocess the text
            transformed_sms = transform_text(input_sms)

            # 2. Vectorize the text
            vector_input = tfidf.transform([transformed_sms])

            # 3. Create and scale the new numerical feature
            num_characters = len(re.sub(r'[^a-zA-Z0-9]', '', input_sms))
            numerical_features = [[num_characters]]
            numerical_features_scaled = scaler.transform(numerical_features)

            # 4. Combine vectorized text with the new feature
            combined_input = hstack([vector_input, numerical_features_scaled])

            # 5. Predict the result
            prediction = model.predict(combined_input.toarray())[0]

            # 6. Display the result with improved styling
            st.divider()
            if prediction == 1:
                st.markdown(
                    "<h2 style='color:red; text-align:center;'>Spam! 😠</h2>",
                    unsafe_allow_html=True
                )
                st.balloons()
            else:
                st.markdown(
                    "<h2 style='color:green; text-align:center;'>Not Spam 😊</h2>",
                    unsafe_allow_html=True
                )
        else:
            st.warning("Please enter a message to classify.")

# Display a success message after all the initial tasks are complete.
if tfidf and model and scaler:
    st.success("App is ready! Try entering a message.")
