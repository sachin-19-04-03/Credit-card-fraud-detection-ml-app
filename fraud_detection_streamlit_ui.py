import streamlit as st
import requests

st.set_page_config(page_title='Fraud Detection Dashboard', layout='centered')

st.title('💳 Credit Card Fraud Detection')
st.write('Enter transaction details and get a real-time fraud prediction from your Flask API.')

# Basic inputs
amount = st.number_input('Transaction Amount', min_value=0.0, value=5000.0)
time_val = st.number_input('Transaction Time', min_value=0.0, value=10000.0)

st.subheader('PCA Features (V1 to V28)')
values = {}
cols = st.columns(4)
for i in range(1, 29):
    with cols[(i-1) % 4]:
        values[f'V{i}'] = st.number_input(f'V{i}', value=0.0, key=f'v{i}')

if st.button('Predict Fraud'):
    payload = {
        'Time': time_val,
        'Amount': amount,
        **values
    }

    try:
        response = requests.post('http://127.0.0.1:5000/predict', json=payload)
        result = response.json()

        st.success('Prediction completed successfully!')
        st.metric('Fraud Score', result.get('fraud_score', 0))
        st.write('### Prediction:', result.get('prediction', 'Unknown'))
        st.write('Fraud Flag:', result.get('fraud_flag', 0))

    except Exception as e:
        st.error(f'API Error: {e}')

st.caption('Make sure your Flask API is running on http://127.0.0.1:5000')
