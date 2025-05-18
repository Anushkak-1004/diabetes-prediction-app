from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler once when app starts
model = pickle.load(open(r'D:\Disease prediction project\model\trained_model.pkl', 'rb'))
scaler = pickle.load(open(r'D:\Disease prediction project\model\scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template(r'D:\Disease prediction project\templates\index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        input_data = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        result = "Positive for disease" if prediction == 1 else "Negative for disease"
        return render_template(r'D:\Disease prediction project\templates\results.html', prediction_text=result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    # Make sure folders exist
    os.makedirs('model', exist_ok=True)
    app.run(debug=True)
