from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('train_model.pkl')

@app.route('/', methods=['POST'])

def predict():
    # Get input data from request
    data = request.get_json()

    # Extract features from input data
    features = preprocess_data(data)

    # Make prediction
    prediction = model.predict(features.reshape(1,-1))
    prediction_list = prediction.tolist()

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
    
    gender_mapping = {'Male': 0, 'Female': 1}  
    subscription_mapping = {'Basic': 0, 'Standard': 1, 'Premium': 2}  
    contract_mapping = {'Monthly': 0, 'Quarterly': 1, 'Annual': 2}
    gender = gender_mapping[data['gender']]
    subscription = subscription_mapping[data['subscription']]
    contract = contract_mapping[data['contract']]

    processed_data = np.array([age,contract, delay, frequency, gender, interaction, spend, subscription, support, tenure])

    print(processed_data)

    return processed_data

if __name__ == '__main__':
    app.run(debug=True, port=5000)