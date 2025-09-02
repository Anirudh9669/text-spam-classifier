import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Set page configuration for a wider, cleaner layout
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📨",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# Load resources with caching for performance
# This decorator ensures the models are only loaded once,
# which is crucial for a fast app on Streamlit Cloud.
@st.cache_resource
def load_models():
    """Loads and caches the model and vectorizer from pickle files."""
    try:
        tfidf_model = pickle.load(open('vectorizer.pkl', 'rb'))
        spam_model = pickle.load(open('model.pkl', 'rb'))
        return tfidf_model, spam_model
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the same directory.")
        return None, None


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
tfidf, model = load_models()

if tfidf and model:
    input_sms = st.text_area("Enter the message here:", height=150)

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button('Predict'):
            # 1. Preprocess the text
            transformed_sms = transform_text(input_sms)

            # 2. Vectorize the text
            vector_input = tfidf.transform([transformed_sms])

            # 3. Predict the result
            prediction = model.predict(vector_input)[0]

            # 4. Display the result with improved styling
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

# Display a success message after all the initial tasks are complete.
if tfidf and model:
    st.success("App is ready! Try entering a message.")
