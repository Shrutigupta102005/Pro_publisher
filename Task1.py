import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from flask import Flask, request, jsonify, render_template

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Load Dataset
def load_dataset():
    # Load your dataset from the downloaded file
    file_path = 'path_to_your_downloaded/research_papers.csv'  # Update this path
    data = pd.read_csv(file_path)

    # Preprocess the text
    data['text'] = data['text'].apply(preprocess_text)
    return data

# Feature Extraction
def extract_features(data):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(data['text']).toarray()
    return X, vectorizer

# Train Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print(classification_report(y_test, y_pred))
    return model

# Flask Web Application
app = Flask(__name__)

# Load the dataset and train the model
data = load_dataset()
X, vectorizer = extract_features(data)
model = train_model(X, data['label'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    processed_text = preprocess_text(text)
    features = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(features)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
