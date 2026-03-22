from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import joblib
from ml_model import estimate_water_quality_and_disease, predict_risk
from ml_model import INDIAN_STATES, MONTHS
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load the model, kmeans, and numeric transformer
try:
    model_data = joblib.load('waterborne_disease_risk_model.pkl')
    model = model_data['model']
    kmeans = model_data['kmeans']
    numeric_transformer = model_data['numeric_transformer']
except FileNotFoundError:
    print("Error: Model file not found.")
    exit(1)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', states=INDIAN_STATES, months=MONTHS)


@app.route('/predict', methods=['POST'])
def predict():
    state = request.form.get('state')
    month = request.form.get('month')
    year = request.form.get('year')

    if not state or state not in INDIAN_STATES:
        flash('Please select a valid state.')
        return redirect(url_for('index'))
    if not month or month not in MONTHS:
        flash('Please select a valid month.')
        return redirect(url_for('index'))
    try:
        year = int(year)
        if year < 2000 or year > 2100:
            flash('Please enter a valid year between 2000 and 2100.')
            return redirect(url_for('index'))
    except ValueError:
        flash('Please enter a valid year.')
        return redirect(url_for('index'))

    rainfall = 200
    water_quality, primary_diseases = estimate_water_quality_and_disease(state, month, year, rainfall)
    result = predict_risk(model, kmeans, numeric_transformer, state, month, year, rainfall, water_quality)

    return render_template('result.html',
                           state=state,
                           month=month,
                           year=year,
                           rainfall=rainfall,
                           risk_level=result['risk_level'],
                           risk_probabilities=result['risk_probabilities'],
                           primary_diseases=primary_diseases,
                           water_quality=water_quality,
                           cluster=result['cluster'])


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
