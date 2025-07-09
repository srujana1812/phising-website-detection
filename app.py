from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("phishing_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        data_scaled = scaler.transform([data])
        pred = model.predict(data_scaled)[0]
        result = "Legitimate Website" if pred == 1 else "Phishing Website"
        return render_template("index.html", prediction_text=f"Prediction: {result}")
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
