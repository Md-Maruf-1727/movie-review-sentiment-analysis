"""
IMDB Sentiment Analysis Web App (Flask)

This Flask application allows users to input movie reviews and get sentiment predictions
(positive or negative) in real-time using the trained Logistic Regression model.
"""

# ------------------------------
# Imports
# ------------------------------
from flask import Flask, render_template, request  # Flask web app components
import joblib  # For loading saved models
import os  # For handling file paths

# ------------------------------
# Initialize Flask App
# ------------------------------
app = Flask(__name__, template_folder='my_templates')  # Set template folder for HTML files

# ------------------------------
# Load Model Artifacts
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get base directory
MODEL_DIR = os.path.join(BASE_DIR, "..", "Models")  # Set path to Models folder

# Load saved Logistic Regression model
model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.joblib"))

# Load saved TF-IDF vectorizer
tfidf = joblib.load(os.path.join(MODEL_DIR,"tfidf_vectorizer.joblib"))

# Load saved Label Encoder
lb = joblib.load(os.path.join(MODEL_DIR,"label_encoder.joblib"))

# ------------------------------
# Home Route
# ------------------------------
@app.route("/", methods=['GET'])
def home():
    """
    Renders the home page with input form for movie review.
    """
    return render_template("index.html")

# ------------------------------
# Predict Route
# ------------------------------
@app.route("/predict", methods=['POST'])
def predict():
    """
    Accepts user input from form, transforms it using TF-IDF, 
    predicts sentiment using Logistic Regression, and returns the result.
    """
    # Get review text from form
    review = request.form['review']

    # Transform input using saved TF-IDF vectorizer
    review_vac = tfidf.transform([review])

    # Predict sentiment
    pred = model.predict(review_vac)

    # Decode numeric label back to 'positive' / 'negative'
    sentiment = lb.inverse_transform(pred)[0]

    # Render same page with prediction result
    return render_template("index.html", prediction=sentiment)

# ------------------------------
# Run Flask App
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)  # Run in debug mode for development
