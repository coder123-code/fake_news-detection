# Import required libraries
import pandas as pd
import numpy as np
import re
import string
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Load datasets
df_fake = pd.read_csv(r"C:\Users\swami\Downloads\Fake.csv")
df_true = pd.read_csv(r"C:\Users\swami\Downloads\True.csv")


# Insert target feature
df_fake["class"] = 0
df_true["class"] = 1

# Merge dataframes
df_merge = pd.concat([df_fake, df_true], axis=0)

# Remove unnecessary columns
df = df_merge.drop(["title", "subject", "date"], axis=1)

# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)


# Text processing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# Apply text processing
df["text"] = df["text"].apply(wordopt)

# Define independent and dependent variables
x = df["text"]
y = df["class"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)

# Train models
LR = LogisticRegression()
LR.fit(xv_train, y_train)

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)


# Prediction function
def predict_news(news):
    processed_news = wordopt(news)
    news_vector = vectorization.transform([processed_news])

    lr_pred = LR.predict(news_vector)[0]
    dt_pred = DT.predict(news_vector)[0]
    rfc_pred = RFC.predict(news_vector)[0]

    predictions = {
        "Logistic Regression": "Fake News" if lr_pred == 0 else "Not A Fake News",
        "Decision Tree": "Fake News" if dt_pred == 0 else "Not A Fake News",
        "Random Forest": "Fake News" if rfc_pred == 0 else "Not A Fake News",
    }

    return predictions


# Streamlit interface
st.title("Fake News Detection")
st.write("Enter a news article below to check if it is fake or not.")

# User input
news_input = st.text_area("News Article")

if st.button("Predict"):
    if news_input:
        predictions = predict_news(news_input)
        st.write("Predictions:")
        for model, result in predictions.items():
            st.write(f"{model}: {result}")
    else:
        st.write("Please enter a news article.")



