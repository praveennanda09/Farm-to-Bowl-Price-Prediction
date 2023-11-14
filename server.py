from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import joblib

app = Flask(__name__)

# Load the preprocessor and the trained model
preprocessor = joblib.load('preprocessor.joblib')
model = xgb.XGBRegressor()
model.load_model('pramodel.bin')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the HTML form
    min_price = float(request.form['min_price'])
    max_price = float(request.form['max_price'])
    commodity = request.form['commodity']
    
    # Create a DataFrame from the user input
    user_input = pd.DataFrame({'min_price': [min_price],
                               'max_price': [max_price],
                               'commodity': [commodity]})
    
    # Transform the user input using the preprocessor
    user_input_preprocessed = preprocessor.transform(user_input)
    
    # Make the prediction using the preprocessed user input
    prediction = model.predict(user_input_preprocessed)
    
    # Display the prediction on the result page
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
