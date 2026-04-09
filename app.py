from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("best_fraud_model_v2.pkl")


# ✅ Preprocessing helper
def preprocess_input(data):
    df = pd.DataFrame([data])

    amount = df.get('Amount', pd.Series([0])).iloc[0]
    time_val = df.get('Time', pd.Series([0])).iloc[0]

    df['Hour'] = (time_val // 3600) % 24
    df['log_amount'] = np.log1p(amount)
    df['amount_to_mean'] = amount / 88.35
    df['is_high_amount'] = int(amount > 2500)
    df['is_night'] = int(df['Hour'].iloc[0] in [0,1,2,3,4,5,22,23])

    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])

    expected_cols = [
        'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
        'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
        'V21','V22','V23','V24','V25','V26','V27','V28',
        'Amount','Hour','is_night','log_amount','amount_to_mean','is_high_amount'
    ]

    df = df.reindex(columns=expected_cols, fill_value=0)

    return df


@app.route('/')
def home():
    return jsonify({
        'message': 'Fraud Detection API is running',
        'endpoint': '/predict',
        'method': 'POST'
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # ✅ call preprocessing function
        processed = preprocess_input(data)

        if hasattr(model, 'predict_proba'):
            fraud_score = float(model.predict_proba(processed)[0][1])
        else:
            fraud_score = float(model.predict(processed)[0])

        prediction = 1 if fraud_score >= 0.5 else 0

        return jsonify({
            'fraud_score': round(fraud_score, 4),
            'prediction': 'Fraud' if prediction == 1 else 'Non-Fraud',
            'fraud_flag': prediction
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)