# Waterborne Disease Risk Prediction System

A machine learning-based web application that predicts waterborne disease risks across Indian states using environmental and water quality parameters.

## Features

- **Risk Prediction**: Predicts disease outbreak risk levels (Low, Moderate, High, Severe) based on state, month, year, and water quality parameters
- **Disease Identification**: Identifies potential waterborne diseases (Cholera, Typhoid, Hepatitis A, Giardiasis, Dysentery)
- **Water Quality Analysis**: Analyzes pH, turbidity, dissolved oxygen, coliform count, and temperature
- **K-means Clustering**: Groups similar risk patterns for better prediction accuracy
- **Email Alert System**: (Coming Soon) Subscribe to receive automated alerts for high-risk predictions in your state
- **Interactive Web Interface**: User-friendly Flask-based web application

## Tech Stack

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, Random Forest Classifier, K-means Clustering
- **Data Processing**: pandas, numpy

- **Frontend**: HTML, Tailwind CSS

## Installation

1. Clone the repository:
```bash
git clone https://github.com/khushiyadav03/Water-Borne-Disease-Prediction.git
cd Water-Borne-Disease-Prediction
```

2. Install required dependencies:
```bash
pip install flask pandas joblib scikit-learn numpy
```

3. Train the model (if not already present):
```bash
python ml_model.py
```

## Usage

1. Run the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

## Project Structure

```
├── app.py                              # Flask web application
├── ml_model.py                         # ML model training and prediction logic
├── waterborne_disease_dataset.csv      # Training dataset
├── waterborne_disease_risk_model.pkl   # Trained model file
├── templates/
│   ├── index.html                      # Home page with prediction form
│   └── result.html                     # Results display page
└── static/
    └── style.css                       # Custom styles
```

## Model Details

- **Algorithm**: Random Forest Classifier with K-means Clustering
- **Features**: State, Month, Year, Rainfall, pH, Turbidity, Dissolved Oxygen, Coliform Count, Water Temperature
- **Output**: Risk Level (Low/Moderate/High/Severe) with probability distribution
- **Clustering**: 4 clusters for pattern recognition

## How It Works

1. **Input**: User selects state, month, and year
2. **Water Quality Estimation**: System estimates water quality parameters based on seasonal patterns
3. **Clustering**: Input data is assigned to a cluster based on similarity
4. **Prediction**: Random Forest model predicts risk level and potential diseases
5. **Alert System**: Daily automated checks send email alerts for high-risk predictions

## Future Enhancements

- Email alert system with background scheduler
- SQLite database for user subscriptions
- Historical data visualization
- API endpoints for integration

## Indian States Covered

All 28 Indian states including:
- Andhra Pradesh, Arunachal Pradesh, Assam, Bihar, Chhattisgarh
- Goa, Gujarat, Haryana, Himachal Pradesh, Jharkhand
- Karnataka, Kerala, Madhya Pradesh, Maharashtra, Manipur
- And more...

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Contact

For questions or feedback, please open an issue on GitHub.
