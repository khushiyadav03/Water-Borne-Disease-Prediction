import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
import joblib

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]


def train_model(dataset_path="waterborne_disease_dataset.csv"):
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)

    print(f"Dataset loaded. Shape: {df.shape}")

    X = df.drop(['primary_disease', 'risk_level', 'risk_score'], axis=1)
    y = df['risk_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = ['rainfall_mm', 'pH', 'turbidity_NTU', 'dissolved_oxygen_mg_L',
                        'total_coliform_MPN', 'water_temp_C']
    categorical_features = ['state', 'month', 'year']

    # Preprocessor for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Apply K-means clustering on numerical features
    print("\nApplying K-means clustering...")
    numeric_transformer = StandardScaler()
    X_numeric = numeric_transformer.fit_transform(X[numeric_features])
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_labels = kmeans.fit_predict(X_numeric)

    # Add cluster labels as a new feature
    X['cluster'] = cluster_labels
    X_train['cluster'] = kmeans.predict(numeric_transformer.transform(X_train[numeric_features]))
    X_test['cluster'] = kmeans.predict(numeric_transformer.transform(X_test[numeric_features]))

    # Update categorical features to include cluster
    categorical_features_updated = ['state', 'month', 'year', 'cluster']

    # Updated preprocessor with cluster feature
    preprocessor_updated = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_updated)
        ])

    print("\nTraining Random Forest model with K-means clusters...")
    model = Pipeline(steps=[
        ('preprocessor', preprocessor_updated),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)

    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    risk_level_map = {'low': 0, 'moderate': 1, 'high': 2, 'severe': 3}
    y_test_num = [risk_level_map[y] for y in y_test]
    y_pred_num = [risk_level_map[y] for y in y_pred]
    r2 = r2_score(y_test_num, y_pred_num)
    mse = mean_squared_error(y_test_num, y_pred_num)

    print(f"\nModel Accuracy: {accuracy:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model, K-means, and numeric transformer
    # joblib.dump({
    #     'model': model,
    #     'kmeans': kmeans,
    #     'numeric_transformer': numeric_transformer
    # }, '../waterborne_disease_prediction/waterborne_disease_risk_model.pkl')
    # print("Model, K-means, and numeric transformer saved as 'waterborne_disease_risk_model.pkl'")
    # return model, kmeans, numeric_transformer

    joblib.dump({
        'model': model,
        'kmeans': kmeans,
        'numeric_transformer': numeric_transformer
    }, '../waterborne_disease_prediction/waterborne_disease_risk_model.pkl')
    print("Model saved as 'waterborne_disease_risk_model.pkl'")
    return model, kmeans, numeric_transformer

def predict_risk(model, kmeans, numeric_transformer, state, month, year, rainfall, water_quality):
    input_data = pd.DataFrame({
        'state': [state],
        'month': [month],
        'year': [year],
        'rainfall_mm': [rainfall],
        'pH': [water_quality['pH']],
        'turbidity_NTU': [water_quality['turbidity_NTU']],
        'dissolved_oxygen_mg_L': [water_quality['dissolved_oxygen_mg_L']],
        'total_coliform_MPN': [water_quality['total_coliform_MPN']],
        'water_temp_C': [water_quality['water_temp_C']]
    })

    # Generate cluster label for the input data
    numeric_features = ['rainfall_mm', 'pH', 'turbidity_NTU', 'dissolved_oxygen_mg_L',
                        'total_coliform_MPN', 'water_temp_C']
    input_numeric = numeric_transformer.transform(input_data[numeric_features])
    cluster_label = kmeans.predict(input_numeric)[0]
    input_data['cluster'] = cluster_label

    risk_level = model.predict(input_data)[0]
    risk_probs = model.predict_proba(input_data)[0]
    risk_probabilities = {level: round(prob, 4) for level, prob in zip(model.classes_, risk_probs)}

    return {
        'risk_level': risk_level,
        'risk_probabilities': risk_probabilities,
        'cluster': cluster_label
    }


def estimate_water_quality_and_disease(state, month, year, rainfall=200):
    month_index = MONTHS.index(month)
    is_monsoon = 5 <= month_index <= 8

    ph = np.random.normal(7.2, 0.5)
    turbidity = np.random.normal(4, 2) * (1.3 if is_monsoon else 1.0)
    dissolved_oxygen = np.random.normal(7, 1.5)
    coliform_count = np.random.poisson(100) * (1.5 if is_monsoon else 1.0)
    temperature = np.random.normal(25, 3) + (
        3 if month_index in [3, 4, 5] else -5 if month_index in [10, 11, 0, 1] else 0)

    if rainfall > 200:
        turbidity += rainfall / 100
        ph -= 0.3
        coliform_count += int(rainfall / 2)

    ph = max(5.5, min(9.0, ph))
    turbidity = max(0.5, turbidity)
    dissolved_oxygen = max(2.0, min(12.0, dissolved_oxygen))
    coliform_count = max(0, coliform_count)
    temperature = max(15, min(35, temperature))

    water_quality = {
        "pH": round(ph, 2),
        "turbidity_NTU": round(turbidity, 2),
        "dissolved_oxygen_mg_L": round(dissolved_oxygen, 2),
        "total_coliform_MPN": int(coliform_count),
        "water_temp_C": round(temperature, 2)
    }

    diseases = ["Cholera", "Typhoid Fever", "Hepatitis A", "Giardiasis", "Dysentery"]
    base_probs = [0.2, 0.2, 0.2, 0.2, 0.2]

    probs = base_probs.copy()
    if coliform_count > 500:
        probs[0] += 0.3
        probs[1] += 0.2
    if turbidity > 8:
        probs[4] += 0.2

    total_prob = sum(probs)
    probs = [p / total_prob for p in probs]

    num_diseases = 2
    selected_diseases = np.random.choice(diseases, size=num_diseases, replace=False, p=probs).tolist()

    return water_quality, selected_diseases


def interactive_prediction(model, kmeans, numeric_transformer):
    print("\n=== Interactive Prediction ===")

    print("\nAvailable States:")
    for i, state in enumerate(INDIAN_STATES):
        print(f"{i}: {state}")
    state_idx = int(input("Enter the index of the state: "))
    state = INDIAN_STATES[state_idx]

    print("\nAvailable Months:")
    for i, month in enumerate(MONTHS):
        print(f"{i}: {month}")
    month_idx = int(input("Enter the index of the month: "))
    month = MONTHS[month_idx]

    year = int(input("Enter the year: "))

    water_quality, primary_diseases = estimate_water_quality_and_disease(state, month, year, rainfall=200)

    input_data = pd.DataFrame({
        'state': [state],
        'month': [month],
        'year': [year],
        'rainfall_mm': [200],
        'pH': [water_quality['pH']],
        'turbidity_NTU': [water_quality['turbidity_NTU']],
        'dissolved_oxygen_mg_L': [water_quality['dissolved_oxygen_mg_L']],
        'total_coliform_MPN': [water_quality['total_coliform_MPN']],
        'water_temp_C': [water_quality['water_temp_C']]
    })

    # Generate cluster label for the input data
    numeric_features = ['rainfall_mm', 'pH', 'turbidity_NTU', 'dissolved_oxygen_mg_L',
                        'total_coliform_MPN', 'water_temp_C']
    input_numeric = numeric_transformer.transform(input_data[numeric_features])
    cluster_label = kmeans.predict(input_numeric)[0]
    input_data['cluster'] = cluster_label

    risk_level = model.predict(input_data)[0]
    risk_probs = model.predict_proba(input_data)[0]
    risk_probabilities = {level: prob for level, prob in zip(model.classes_, risk_probs)}

    print("\n=== Prediction Results ===")
    print(f"State: {state}")
    print(f"Month: {month}")
    print(f"Year: {year}")
    print(f"Predicted Risk Level: {risk_level}")
    print("Risk Level Probabilities:")
    for level, prob in risk_probabilities.items():
        print(f"  {level}: {prob:.4f}")
    print(f"Predicted Primary Diseases: {', '.join(primary_diseases)}")
    print("Water Quality Parameters:")
    for key, value in water_quality.items():
        print(f"  {key}: {value}")
    print(f"Assigned Cluster: {cluster_label}")


if __name__ == "__main__":
    print("Waterborne Disease Risk Prediction Model with K-means Clustering")
    print("=======================================")

    print("No existing model found. Training new model...")
    model, kmeans, numeric_transformer = train_model('waterborne_disease_dataset.csv')

    interactive_prediction(model, kmeans, numeric_transformer)