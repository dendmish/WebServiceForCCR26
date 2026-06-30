from flask import Flask, request, jsonify, render_template

import pandas as pd
import joblib

app = Flask(__name__)

# Загрузка модели и списка признаков
model = joblib.load('churn_pipeline (1).pkl')
features = joblib.load('features.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/report')
def report():

    return render_template(
        'report.html'
    )

@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.json

        input_df = pd.DataFrame([data])

        input_df = input_df[features]

        prediction = int(
            model.predict(input_df)[0]
        )

        probability = float(
            model.predict_proba(input_df)[0][1]
        )

        return jsonify({
            "prediction": prediction,
            "churn_probability": round(probability, 4)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400

@app.route('/form_predict', methods=['POST'])
def form_predict():

    try:

        input_df = pd.DataFrame([{
            'Payment Delay': int(request.form['payment_delay']),
            'Support Calls': int(request.form['support_calls']),
            'Tenure': int(request.form['tenure']),
            'Usage Frequency': int(request.form['usage_frequency']),
            'Gender_Male': int(request.form['gender_male'])
        }])

        prediction = int(
            model.predict(input_df)[0]
        )

        probability = float(
            model.predict_proba(input_df)[0][1]
        )

        return render_template(
            'index.html',
            prediction=prediction,
            probability=round(probability, 4)
        )

    except Exception as e:

        return str(e)

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )

    