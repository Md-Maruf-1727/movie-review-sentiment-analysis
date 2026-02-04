from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, render_template='templates')

model = joblib.load("../Models/logistic_model.joblib")
tfidf = joblib.load("../Models/tfidf_vectorizer.joblib")
lb = joblib.load("../Models/label_encoder.joblib")

app.route("/", methods= ['GET'])
app.route("/predict", methods = ['POST'])




if __name__ == "__main__":
    app.run(debug=True)