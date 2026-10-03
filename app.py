# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
Product_Store_Sales_Total_api = Flask("Sales total prediction")

# Load the trained Boston housing model
model1 = joblib.load("SuparKart_sale_prediction_v1.0.joblib")

# Define a route for the home page (GET request)
@Product_Store_Sales_Total_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the Sales prediction based on product & store API!"

# Define an endpoint for single property prediction (POST request)
@Product_Store_Sales_Total_api.post('/v1/rental')
def Product_Store_Sales_Total():
    """
    This function handles POST requests to the '/v1/rental' endpoint.
    It expects a JSON payload containing property details and returns
    the predicted rental price as a JSON response.
    """
    # Get the JSON data from the request body
    property_data = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        'Product ID': property_data['Product_Id'],
        'Product weight': property_data['Product_Weight'],
        'Sugar Content': property_data['Product_Sugar_Content'],
        'Allocated area': property_data['Product_Allocated_Area'],
        'Product type': property_data['Product_Type'],
        'Product MRP': property_data['Product_MRP'],
        'Store ID': property_data['Store_Id'],
        'Store establishment year': property_data['Store_Establishment_Year'],
        'Store Size': property_data['Store_Size'],
        'SStore Loc City type': property_data['Store_Location_City_Type'],
        'Store type': property_data['Store_Type']

    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction (get log_sale)
    predicted_log_sale = model1.predict(input_data)[0]

    # Calculate actual sale
    predicted_sale = np.exp(predicted_log_sale)

    # Convert predicted_sale to Python float
    predicted_sale = round(float(predicted_sale), 2)
    # The conversion above is needed as we convert the model prediction (log price) to actual price using np.exp, which returns predictions as NumPy float32 values.
    # When we send this value directly within a JSON response, Flask's jsonify function encounters a datatype error

    # Return the actual price
    return jsonify({'Predicted sale (in dollars)': predicted_sale})


# Define an endpoint for batch prediction (POST request)
@Product_Store_Sales_Total_api.post('/v1/rentalbatch')
def Product_Store_Sales_Total():
    """
    This function handles POST requests to the '/v1/rentalbatch' endpoint.
    It expects a CSV file containing property details for multiple properties
    and returns the predicted rental prices as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all properties in the DataFrame (get log_sale)
    predicted_log_sale = model.predict(input_data).tolist()

    # Calculate actual sales
    predicted_sale = [round(float(np.exp(log_sale)), 2) for log_sale in predicted_log_sale]

    # Create a dictionary of predictions with property IDs as keys
    property_ids = input_data['id'].tolist()  # Assuming 'id' is the property ID column
    output_dict = dict(zip(property_ids, predicted_sale))  # Use actual prices

    # Return the predictions dictionary as a JSON response
    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    Product_Store_Sales_Total_api.run(debug=True)
