# FeelFusion: Brand Sentiment Analyzer

FeelFusion is a web-based sentiment analysis tool that predicts whether a review is positive (👍) or negative (👎) using machine learning models. It supports both single-review analysis and batch processing of reviews from uploaded files (CSV, Excel, or text).

## Table of Contents

1. [Overview](#overview)
2. [Dataset Source](#dataset-source)
3. [Key Columns](#key-columns)
4. [Project Structure](#project-structure)
5. [Key Features and Methods](#key-features-and-methods)
6. [Technologies Used](#technologies-used)
7. [Installation](#installation)
8. [Usage](#usage)
   

## Overview

FeelFusion is a powerful tool designed to help brands understand customer opinions by analyzing reviews. Using advanced sentiment analysis techniques, FeelFusion provides valuable insights to improve customer satisfaction and reputation management.

## Dataset Source
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mahmoudshaheen1134/amazon-alexa-reviews-dataset/data).

## Key Columns

- **Rating** : The numerical rating given by the customer, typically on a scale from 1 to 5, where 1 is the lowest rating and 5 is the highest.

- **Date** : The date when the review was posted.

- **Variation** : The specific variant or version of the product being reviewed (e.g., color, finish).

- **Verified_reviews** : The actual text of the review written by the customer.

- **Feedback** : Indicates whether the review was marked positive or negative by other users (usually a binary value, with 1 meaning "Positive" and 0 meaning "Negative").

## Project Structure

```
FeelFusion-Brand_Sentiment_Analyzer/
├── app/
│   ├── app.py
│  
├── data/
│   ├── feature_engineered_data/
│   │   └── amazon_alexa.tsv              # Processed data for model building
│   └── raw_data/
│       └── amazon_alexa.tsv              # Source: Kaggle
│
├── Models/
│   ├── countvectorizer.pkl               # CountVectorizer
│   ├── model_rf.pkl                      # Trained Random Forest model
│   ├── model_xgb.pkl                     # Trained  XGBoost model
│   └── scaler.pkl                        # StandardScaler
│
├── notebook/
│   ├── 1_data_preprocessing.ipynb        # Data checks & cleaning
│   ├── 2_EDA.ipynb                       # Exploratory Data Analysis
│   └── 3_model_building.ipynb            # Model training & evaluation
│
└── requirements.txt                      # Required libraries
```

## Key Features and Methods

1. **Data Preprocessing** (`1_data_preprocessing.ipynb`):
   - Handled null values and inconsistencies in the dataset.
   - Encoded categorical variables and normalized numerical data.

2. **Exploratory Data Analysis** (`2_EDA.ipynb`):
   - Investigated distributions and key patterns in customer reviews.
   - Analyzed review lengths and sentiment distributions.
   - Visualized trends over time and variations in product feedback.

4. **Model Building** (`3_model_building.ipynb`):
   - Created new features such as review length and sentiment scores. 
   - Encodes categorical features and scales numerical data.
   - Trained and evaluated two models: Random Forest Classifier, XGBoost Classifier, DecisionTreeClassifier.

## Technologies Used

- **Python Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, nltk
- **Jupyter Notebook:** For interactive data exploration and analysis
- **Web Framework:** **Streamlit** For building and deploying the web application.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Santheeppa/FeelFusion-Brand_Sentiment_Analyzer.git
   cd FeelFusion-Brand_Sentiment_Analyzer
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
   ```bash
   streamlit run app.py
   ```

