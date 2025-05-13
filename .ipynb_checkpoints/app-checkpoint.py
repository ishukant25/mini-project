# app.py

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os

# Initialize Flask app
app = Flask(__name__)

# Load models and preprocessor
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
densenet128_model = tf.keras.models.load_model(os.path.join(models_dir, 'densenet128_model.h5'))
rf_model = joblib.load(os.path.join(models_dir, 'enhanced_rf_model.pkl'))
preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))

# Home route
@app.route('/')
def home():
    """
    Render the home page with the input form.
    """
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle form submission, preprocess input data, make predictions, and render the result page.
    """
    # Get input data from the form
    input_data = {
        'radius_mean': float(request.form['radius_mean']),
        'texture_mean': float(request.form['texture_mean']),
        'perimeter_mean': float(request.form['perimeter_mean']),
        'area_mean': float(request.form['area_mean']),
        'smoothness_mean': float(request.form['smoothness_mean']),
        'compactness_mean': float(request.form['compactness_mean']),
        'concavity_mean': float(request.form['concavity_mean']),
        'concave_points_mean': float(request.form['concave_points_mean']),
        'symmetry_mean': float(request.form['symmetry_mean']),
        'fractal_dimension_mean': float(request.form['fractal_dimension_mean']),
        'radius_se': float(request.form['radius_se']),
        'texture_se': float(request.form['texture_se']),
        'perimeter_se': float(request.form['perimeter_se']),
        'area_se': float(request.form['area_se']),
        'smoothness_se': float(request.form['smoothness_se']),
        'compactness_se': float(request.form['compactness_se']),
        'concavity_se': float(request.form['concavity_se']),
        'concave_points_se': float(request.form['concave_points_se']),
        'symmetry_se': float(request.form['symmetry_se']),
        'fractal_dimension_se': float(request.form['fractal_dimension_se']),
        'radius_worst': float(request.form['radius_worst']),
        'texture_worst': float(request.form['texture_worst']),
        'perimeter_worst': float(request.form['perimeter_worst']),
        'area_worst': float(request.form['area_worst']),
        'smoothness_worst': float(request.form['smoothness_worst']),
        'compactness_worst': float(request.form['compactness_worst']),
        'concavity_worst': float(request.form['concavity_worst']),
        'concave_points_worst': float(request.form['concave_points_worst']),
        'symmetry_worst': float(request.form['symmetry_worst']),
        'fractal_dimension_worst': float(request.form['fractal_dimension_worst'])
    }

    # Preprocess the input data
    input_df = pd.DataFrame([input_data])
    preprocessed_data = preprocessor.transform(input_df)

    # Make predictions
    densenet128_prediction = (densenet128_model.predict(preprocessed_data) > 0.5).astype(int)
    rf_prediction = rf_model.predict(preprocessed_data)

    # Render the result page
    return render_template('result.html', 
                           densenet128_prediction=densenet128_prediction[0],
                           rf_prediction=rf_prediction[0])

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)