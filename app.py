from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('train_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['POST'])

def predict():
    # Get input data from request
    data = request.get_json()

    # Extract features from input data
    features = preprocess_data(data)

    # Make prediction
    print(features.reshape(1, -1))
    prediction = model.predict(features.reshape(1, -1))
    prediction_list = prediction.tolist()
    print(prediction_list)


    # Return prediction result
    return jsonify({'churn_prediction': prediction_list})

def preprocess_data(data):
    # Convert string values to appropriate data types
    age = int(data['age'])
    delay = int(data['delay'])
    frequency= int(data['frequency'])
    interaction = int(data['interaction'])
    spend = int(data['spend'])
    support= int(data['support'])
    tenure = int(data['tenure'])

    # Encode categorical variables
    
    gender_mapping = {'Male': 1, 'Female': 0}  
    subscription_mapping = {'Basic': 0, 'Standard': 2, 'Premium': 1}  
    contract_mapping = {'Monthly': 1, 'Quarterly': 12, 'Annual': 0}
    gender = gender_mapping[data['gender']]
    subscription = subscription_mapping[data['subscription']]
    contract = contract_mapping[data['contract']]

    processed_data = np.array([[age,gender, tenure, frequency, support, delay, subscription,contract, spend, interaction]])

    scaled_data = scaler.transform(processed_data)
    print(scaled_data)

    return scaled_data

if __name__ == '__main__':
    app.run(debug=True, port=5000)