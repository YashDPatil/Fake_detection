from flask import Flask, render_template, request
import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import re

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__, template_folder='template')

# Load the dataset
dataset = pd.read_csv(r'C:\Users\yash\Desktop\fake_detect\amazon_reviews_2019.csv', encoding='latin1')

# Preprocessing and feature extraction
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove punctuation and non-alphabet characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and meaningless words
    filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 1]

    # Join the tokens back into a single string
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(dataset['review_text'].apply(preprocess_text))
y_train = dataset['review_type']

# Define the models and parameter grid for GridSearchCV
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC()
}

param_grid = {
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'Logistic Regression': {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

# Find the best model and its parameters using GridSearchCV
best_model = None
best_accuracy = 0.0

for model_name, model in models.items():
    clf = GridSearchCV(model, param_grid[model_name], cv=3)
    clf.fit(X_train, y_train)

    # Get the best model and its accuracy
    if clf.best_score_ > best_accuracy:
        best_model = clf.best_estimator_
        best_accuracy = clf.best_score_

# Define the Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
    review_text = request.form['review_text']
    
    # Check if the input is empty or contains only meaningless words or random characters
    if not review_text or all(word not in stop_words for word in review_text.split()):
        return render_template('index.html', prediction='Entered Review May be Fake/Invalid')
        
    cleaned_text = preprocess_text(review_text)

    # Transform input data using the same CountVectorizer
    X_input = vectorizer.transform([cleaned_text])

    # Make the prediction
    prediction = best_model.predict(X_input)[0]

    return render_template('index.html', prediction=prediction, accuracy=best_accuracy)


if __name__ == '__main__':
    app.run(debug=True)
