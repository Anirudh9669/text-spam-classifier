import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy.sparse import hstack
import warnings

# Suppress the annoying UserWarning from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)

# --- NLTK Downloads (required for first run) ---
# Use st.spinner to show a loading message during downloads
with st.spinner('Downloading NLTK data... This will only happen once.'):
    try:
        nltk.data.find('corpora/stopwords')
    except:  # Using a general except to avoid the import error
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except:  # Using a general except to avoid the import error
        nltk.download('punkt')

# --- Helper Function for Text Transformation ---
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


# --- Model Loading with Streamlit Caching ---
# The @st.cache_resource decorator loads the models only once, improving performance
# This is crucial for deployment to avoid reloading on every user interaction.
@st.cache_resource
def load_models():
    try:
        tfidf_model = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return tfidf_model, model, scaler
    except FileNotFoundError:
        st.error(
            "Error: Model files not found. Please run the Jupyter notebook to train the model and generate the .pkl files.")
        st.stop()


# Load all models once
tfidf, model, scaler = load_models()

# --- Streamlit UI/UX ---
# Custom CSS for a modern, centered layout and color scheme
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .centered-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        padding: 1rem;
    }
    .title-box {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 2rem;
    }
    h1 {
        font-size: 2.5rem;
        color: #333;
    }
    .subtitle {
        color: #555;
    }
    .result-box {
        margin-top: 2rem;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        color: white;
    }
    .result-spam {
        background-color: #e53935; /* Red for Spam */
    }
    .result-not-spam {
        background-color: #43a047; /* Green for Not Spam */
    }
    </style>
""", unsafe_allow_html=True)

# Main container for centering content
with st.container():
    st.markdown("<div class='centered-container'>", unsafe_allow_html=True)

    # Title and subtitle box
    st.markdown("<div class='title-box'>", unsafe_allow_html=True)
    st.markdown("<h1>Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>A machine learning tool to detect unwanted messages</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Text area for user input
    input_sms = st.text_area("Enter the message here", height=150)

    # Predict button
    if st.button('Classify'):
        if input_sms:
            # Show a spinner while processing
            with st.spinner('Analyzing...'):
                # 1. Preprocess the text
                transformed_sms = transform_text(input_sms)

                # 2. Vectorize the text
                vector_input = tfidf.transform([transformed_sms])

                # 3. Create and scale the new numerical feature
                num_characters = len(input_sms)
                numerical_features = [[num_characters]]
                numerical_features_scaled = scaler.transform(numerical_features)

                # 4. Combine vectorized text with the new feature
                combined_input = hstack([vector_input, numerical_features_scaled])

                # 5. Convert to dense array
                combined_input_dense = combined_input.toarray()

                # 6. Predict
                result = model.predict(combined_input_dense)[0]

            # 7. Display the result in a clean box
            if result == 1:
                st.markdown("<div class='result-box result-spam'><h1>❌ Spam</h1></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-box result-not-spam'><h1>✅ Not Spam</h1></div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a message to classify.")

    st.markdown("</div>", unsafe_allow_html=True)
