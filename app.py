from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load pre-trained scaler and models
scaler = joblib.load('models/scaler.pkl')
model_lr = joblib.load('models/model_lr.pkl')
model_rf = joblib.load('models/model_rf.pkl')
model_xgb = joblib.load('models/model_xgb.pkl')
model_svm = joblib.load('models/model_svm.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input values from form
            V10 = float(request.form['V10'])
            V12 = float(request.form['V12'])
            V14 = float(request.form['V14'])
            V16 = float(request.form['V16'])
            V17 = float(request.form['V17'])
            Amount = float(request.form['Amount'])
        except ValueError:
            return render_template('predict.html', error="Please enter valid numeric input.")

        features = np.array([[V10, V12, V14, V16, V17, Amount]])
        features_scaled = scaler.transform(features)

        algorithm = request.form['algorithm']
        if algorithm == 'Logistic Regression':
            prediction = model_lr.predict(features_scaled)[0]
        elif algorithm == 'Random Forest':
            prediction = model_rf.predict(features_scaled)[0]
        elif algorithm == 'XGBoost':
            prediction = model_xgb.predict(features_scaled)[0]
        elif algorithm == 'SVM':
            prediction = model_svm.predict(features_scaled)[0]
        else:
            return render_template('predict.html', error="Invalid algorithm selected.")

        label = 'Fraud' if prediction == 1 else 'No Fraud'
        return render_template('result.html', algorithm=algorithm, label=label)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
