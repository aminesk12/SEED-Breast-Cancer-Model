import os
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__, template_folder='template')

# Define numeric columns and form options
NUMERIC_COLUMNS = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']
FORM_OPTIONS = {
    '6th Stage': ['IIA', 'IIB', 'IIC', 'IIIA', 'IIIB', 'IIIC'],
    'T Stage': ['T1', 'T2', 'T3'],
    'N Stage': ['N1', 'N2', 'N3'],
    'Grade': ['Well differentiated; Grade I', 'Moderately differentiated; Grade II', 'Poorly differentiated; Grade III', 'Undifferentiated; anaplastic; Grade IV']
}

# ----------------------------
# Function to load necessary files
# ----------------------------
def load_files(model_dir="./model"):
    # Define file paths
    file_paths = {
        'model': os.path.join(model_dir, "best_model.pkl"),
        'info': os.path.join(model_dir, "model_info.txt"),
        'feature_order': os.path.join(model_dir, "feature_order.pkl"),
        'label_encoders': os.path.join(model_dir, "label_encoders.pkl"),
        'scaler': os.path.join(model_dir, "scaler.pkl")
    }

    # Check if all files exist
    for name, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file '{name}' not found: {path}")

    # Load components
    best_model = joblib.load(file_paths['model'])
    with open(file_paths['info'], "r") as f:
        model_info = f.read()
    feature_order = joblib.load(file_paths['feature_order'])
    label_encoders = joblib.load(file_paths['label_encoders'])
    scaler = joblib.load(file_paths['scaler'])

    return best_model, feature_order, label_encoders, scaler, model_info


# ----------------------------
# Load model and components
# ----------------------------
try:
    best_model, feature_order, label_encoders, scaler, model_info = load_files()
    print("Model and components loaded successfully.")
except Exception as e:
    print(f"Error loading model components: {e}")
    raise e


# ----------------------------
# Routes
# ----------------------------

@app.route('/', methods=['GET'])
def home():
    """Render the home page with form options."""
    return render_template(
        'index.html',
        form_options=FORM_OPTIONS,
        numeric_columns=NUMERIC_COLUMNS,
        model_info=model_info
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        if not best_model or not scaler:
            raise ValueError("Model or scaler is not loaded.")

        # Get form data
        data = request.form.to_dict()

        # Log received data
        print("Received Form Data:", data)

        # Create a DataFrame with a single row
        df = pd.DataFrame([data])

        # Convert numeric fields
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Apply label encoding to categorical columns
        for col, encoder in label_encoders.items():
            if col in df.columns:
                # Validate category before encoding
                if df[col].iloc[0] not in encoder.classes_:
                    return jsonify({'error': f"Invalid category for {col}: {df[col].iloc[0]}"}), 400
                df[col] = encoder.transform(df[col])

        # Ensure the DataFrame matches the expected feature order
        df = df[feature_order]

        # Scale the numeric features
        df[NUMERIC_COLUMNS] = scaler.transform(df[NUMERIC_COLUMNS])

        # Make prediction
        prediction = best_model.predict(df)
        prediction_proba = best_model.predict_proba(df)

        # Prepare and return the response
        response = {
            'prediction': 'Dead' if int(prediction[0]) == 1 else 'Alive',
            'probability': float(prediction_proba[0][1]),
            'model_type': model_info
        }


        return jsonify(response)

    except Exception as e:
        # Log the full traceback for debugging
        import traceback
        print("Error during prediction:", traceback.format_exc())
        return jsonify({'error': str(e)}), 400


# ----------------------------
# Run the app
# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)
