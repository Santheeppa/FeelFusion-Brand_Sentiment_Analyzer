import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
from sklearn.preprocessing import MinMaxScaler

# Download NLTK resources
nltk.download('stopwords')

# Load models and vectorizers
@st.cache_resource
def load_models():
    cv = pickle.load(open('../Models/countVectorizer.pkl', 'rb'))
    scaler = pickle.load(open('../Models/scaler.pkl', 'rb'))
    model_rf = pickle.load(open('../Models/model_rf.pkl', 'rb'))
    model_xgb = pickle.load(open('../Models/model_xgb.pkl', 'rb'))
    return cv, scaler, model_rf, model_xgb

cv, scaler, model_rf, model_xgb = load_models()

# Text preprocessing function
def preprocess_text(text):
    ps = PorterStemmer()
    custom_stopwords = set(stopwords.words('english')) - {'not', 'no', 'never'}
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in custom_stopwords]
    return ' '.join(review)

# Define stemmed negative keywords
NEGATIVE_KEYWORDS = {
    'worst', 'bad', 'horribl', 'terribl', 'aw', 'wors', 'disappoint', 'fail',
    'poor', 'useless', 'hate', 'problem', 'wast', 'suck', 'broke', 'return',
    'never', 'no', 'not', 'disgust', 'unreliabl', 'junk', 'trash', 'garbag',
    'stupid', 'crap', 'didnt', 'dont', 'cant', 'couldnt', 'wouldnt', 'wont',
    'shouldnt', 'noway', 'nowhere', 'nope', 'none', 'nobody', 'nothing', 'neither',
    'nor', 'hardly', 'scarcely', 'barely', 'doesnt', 'isnt', 'wasnt', 'arent',
    'werent', 'havent', 'hasnt', 'hadnt'
}

def analyze_review(review):
    processed_review = preprocess_text(review)
    processed_words = set(processed_review.split())
    
    has_negative = any(word in NEGATIVE_KEYWORDS for word in processed_words)
    vectorized_review = cv.transform([processed_review])
    
    if has_negative or vectorized_review.sum() == 0:
        return 0, 0, processed_review  # Both models return negative
    else:
        scaled_review = scaler.transform(vectorized_review.toarray())
        rf_pred = model_rf.predict(scaled_review)[0]
        xgb_pred = model_xgb.predict(vectorized_review)[0]
        return rf_pred, xgb_pred, processed_review

# Streamlit app
st.title('FeelFusion: Brand Sentiment Analyzer')
st.write('Predict if a review is positive (👍) or negative (👎)')

# File upload section
st.header("Batch Processing")
uploaded_file = st.file_uploader("Upload any file with reviews", type=None)
if uploaded_file is not None:
    try:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Read file based on extension
        if file_ext == '.csv':
            df = pd.read_csv(uploaded_file)
        elif file_ext in ('.xls', '.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif file_ext == '.txt':
            text_content = uploaded_file.getvalue().decode("utf-8")
            df = pd.DataFrame({'review': text_content.split('\n')})
        else:
            # Try to read as text file
            try:
                text_content = uploaded_file.getvalue().decode("utf-8")
                df = pd.DataFrame({'review': text_content.split('\n')})
            except:
                st.error("Unsupported file format. Please upload CSV, Excel, or text file.")
                df = None
        
        if df is not None:
            if 'review' not in df.columns:
                # Try to use first column as reviews
                if len(df.columns) > 0:
                    df['review'] = df.iloc[:, 0]
                else:
                    st.error("No recognizable review content found in the file")
            
            results = []
            for index, row in df.iterrows():
                if pd.notna(row['review']) and str(row['review']).strip() != '':
                    rf_pred, xgb_pred, processed = analyze_review(str(row['review']))
                    results.append({
                        'Original Review': row['review'],
                        'Processed Text': processed,
                        'RF Prediction': '👍 Positive' if rf_pred == 1 else '👎 Negative',
                        'XGB Prediction': '👍 Positive' if xgb_pred == 1 else '👎 Negative'
                    })
            
            if results:
                result_df = pd.DataFrame(results)
                st.subheader("Analysis Results")
                st.dataframe(result_df)
                
                # Add download button
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='review_predictions.csv',
                    mime='text/csv'
                )

                # Show prediction percentages
                st.subheader("Prediction Distribution")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Random Forest Model**")
                    rf_pos = (result_df['RF Prediction'] == '👍 Positive').mean() * 100
                    st.write(f"Positive: {rf_pos:.2f}%")
                    st.write(f"Negative: {100 - rf_pos:.2f}%")

                with col2:
                    st.markdown("**XGBoost Model**")
                    xgb_pos = (result_df['XGB Prediction'] == '👍 Positive').mean() * 100
                    st.write(f"Positive: {xgb_pos:.2f}%")
                    st.write(f"Negative: {100 - xgb_pos:.2f}%")
            else:
                st.warning("No valid reviews found in the file")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Single review section
st.header("Single Review Analysis")
review = st.text_area('Enter your review here:')

if st.button('Analyze Review'):
    if review.strip() == '':
        st.warning("Please enter a review to analyze")
    else:
        rf_pred, xgb_pred, processed_review = analyze_review(review)
        
        st.subheader('Predictions:')
        col1, col2 = st.columns(2)
        with col1:
            st.write('Random Forest Prediction:', '👍 Positive' if rf_pred == 1 else '👎 Negative')
        with col2:
            st.write('XGBoost Prediction:', '👍 Positive' if xgb_pred == 1 else '👎 Negative')
        
        st.subheader('Processed Text:')
        st.write(processed_review)

# Add some explanations
st.sidebar.header('About')
st.sidebar.info('FeelFusion: Brand Sentiment Analyzer helps brands understand customer opinions by analyzing reviews. This app uses two powerful machine learning models to classify reviews as positive or negative and provides valuable insights to improve customer satisfaction and reputation management.')
st.sidebar.subheader('Model Information')
st.sidebar.write('Trained Models:')
st.sidebar.write('- Random Forest Classifier')
st.sidebar.write('- XGBoost Classifier')
st.sidebar.write('Accuracy: ~94%')