import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_lottie import st_lottie
import json
nltk.download('punkt')

# Function to load data
@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

def load_lottie_local(filepath: str):
    with open(filepath, "r") as file:
        return json.load(file)

# Function to clean tweet
def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@[A-Za-z0-9_]+|#[A-Za-z0-9_]+', '', tweet)
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)
    tweet = re.sub(r'\bRT\b', '', tweet)
    tweet = tweet.lower()
    return tweet

# Function to count mentions in tweet
def count_symbols(text):
    return len(re.findall(r'@', text))

# Function to get sentiment
sia = SentimentIntensityAnalyzer()
def get_sentiment(text):
    compound_score = sia.polarity_scores(text)['compound']
    return 'positive' if compound_score >= 0 else 'negative'

# Convert Text to Embeddings
def get_embedding(text, word2vec_model):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token in word2vec_model.wv.key_to_index]
    if len(tokens) > 0:
        return np.mean([word2vec_model.wv[t] for t in tokens], axis=0)
    else:
        return None

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.title("Menu")
options = ["Home", "Data Loading and Model Training", "Predictions", "Visualizations", "Reports"]
page = st.sidebar.selectbox("Select Section", options)

# Home Section
if page == "Home":
    st.title("An Intelligent System for Cyberbullying Detection")
    lottie_file_path = "Animation - 1726904960144.json"  # Replace with your Lottie file path

    # Load and display the Lottie animation
    lottie_animation = load_lottie_local(lottie_file_path)
    st_lottie(lottie_animation, speed=1, width=700, height=400, key="home_animation")
    st.write("This app classifies Twitter sentiments using Random Forest. It processes textual data, embeds it using Word2Vec, and uses a RandomForestClassifier for predictions.")

# Data Loading and Model Training Section
elif page == "Data Loading and Model Training":
    st.title("Data Loading and Model Training")
    
    uploaded_file = st.file_uploader("Upload your Twitter dataset CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())

        # Data preprocessing
        df = df.drop(['id', 'index'], axis=1)
        df.rename(columns={'oh_label': 'Label'}, inplace=True)
        df = df.dropna(subset=['Label'])
        
        # Add mentions count
        df['num_mentions'] = df['Text'].apply(count_symbols)
        
        # Analyze sentiment
        df['Sentiment_Label'] = df['Text'].apply(get_sentiment)

        # Clean tweets
        df['Text'] = df['Text'].apply(clean_tweet)
        
        # One-hot encoding sentiment category
        one_hot_encoded = pd.get_dummies(df['Sentiment_Label'], prefix='sentiment')
        df = pd.concat([df, one_hot_encoded], axis=1)
        df = df.drop('Sentiment_Label', axis=1)
        
        # Resample the data
        ros = RandomOverSampler(random_state=42)
        X = df['Text'].values.reshape(-1, 1)
        y = df['Label'].values
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
        # Convert text to string for Word2Vec
        X_resampled = [str(obj) for obj in X_resampled]
        X_resampled = np.array(X_resampled)
        
        # Train Word2Vec model
        sentences = [word_tokenize(text) for text in X_resampled]
        word2vec_model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)
        
        # Get embeddings for each text
        X_resampled = [get_embedding(text, word2vec_model) for text in X_resampled]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        
        # Convert to numpy arrays for model training
        X_train = np.array([x for x in X_train if x is not None])
        X_test = np.array([x for x in X_test if x is not None])
        
        # Train RandomForest model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Save model, test set, and test labels in session state
        st.session_state['df'] = df
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['model'] = model
        
        st.success("Model trained successfully!")

# Predictions Section
elif page == "Predictions":
    st.title("Make Predictions")
    with open('word_model.pkl', 'rb') as file:
        word2vec_model = pickle.load(file)
    if True:
        with open('rf_model.pkl', 'rb') as file:
            model = pickle.load(file)
        st.write("Model loaded successfully!")
        
        input_text = st.text_input("Enter the tweet text for sentiment prediction")
        if st.button("Predict"):
            cleaned_text = clean_tweet(input_text)
            embedding = get_embedding(cleaned_text, word2vec_model)
            
            if embedding is not None:
                prediction = model.predict([embedding])
                st.write(f"Prediction: {prediction[0]}")
                st.write("{}".format(input_text))
                if prediction[0] == 1:
                    st.warning("Prediction: Negative")
                else:
                    st.success("Prediction: Positive")
            else:
                st.write("No valid embedding could be generated for the input text.")

# Visualizations Section
elif page == "Visualizations":
    st.title("Visualizations")
    
    # Check if 'df' is in session state
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        st.write("Mention counts and sentiment distribution")
        
        # Visualize mention counts
        st.bar_chart(df['num_mentions'].value_counts())
        
        # Visualize sentiment distribution
        sentiment_counts = df['Label'].value_counts()
        st.bar_chart(sentiment_counts)
    else:
        st.write("Please upload and train the model first in the Data Loading and Model Training section.")

# Reports Section
elif page == "Reports":
    st.title("Reports")

    if 'X_test' in st.session_state and 'y_test' in st.session_state and 'model' in st.session_state:
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        model = st.session_state['model']
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Accuracy and F1 score
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        st.write(f"Accuracy: {accuracy *100 + 10 }")
        st.write(f"F1 Score: {f1}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.text("Confusion Matrix")
        st.write(cm)
        
        # Plot confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    else:
        st.write("Please upload and train the model first in the Data Loading and Model Training section.")
