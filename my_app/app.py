from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__, template_folder='my_templates')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "Models")

model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.joblib"))
tfidf = joblib.load(os.path.join(MODEL_DIR,"tfidf_vectorizer.joblib"))
lb = joblib.load(os.path.join(MODEL_DIR,"label_encoder.joblib"))

@app.route("/", methods= ['GET'])
def home():
    return render_template("index.html")


@app.route("/predict", methods= ['POST'])
def predict():
    review = request.form['review']

    review_vac = tfidf.transform([review])
    pred = model.predict(review_vac)
    sentiment = lb.inverse_transform(pred)[0]

    return render_template("index.html", prediction=sentiment)

if __name__ == "__main__":
    app.run(debug=True)