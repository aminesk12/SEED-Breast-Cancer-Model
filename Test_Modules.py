import pandas as pd
import joblib

# Load pipeline components
feature_order = joblib.load("model/feature_order.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
scaler = joblib.load("model/scaler.pkl")
model = joblib.load("model/best_model.pkl")

# Sample input
sample_data = {
    'Age': 20 ,
    'Tumor Size': 10,
    'Regional Node Examined': 4,
    'Reginol Node Positive': 1,
    'Survival Months': 4,
    '6th Stage': 'IB',
    'T Stage': 'T2',
    'N Stage': 'N1',
    'Grade': 'Moderately differentiated; Grade I'
}

# Convert to DataFrame
df = pd.DataFrame([sample_data])

# Encode categorical columns
for col, encoder in label_encoders.items():
    df[col] = encoder.transform(df[col])

# Reorder columns
df = df[feature_order]

# Scale numeric columns
numeric_columns = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']
df[numeric_columns] = scaler.transform(df[numeric_columns])

# Predict
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

print(f"Prediction: {prediction[0]}")
print(f"Probability of Death: {prediction_proba[0][1]:.4f}")
