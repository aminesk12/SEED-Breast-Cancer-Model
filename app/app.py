import os
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__, template_folder='template')

# Load necessary files
def load_files():
    """
    Load the model and its info from disk.
    """
    model_dir = "./model"
    model_file = os.path.join(model_dir, "best_pipeline.pkl")
    model_info_file = os.path.join(model_dir, "model_info.txt")

    # Ensure necessary files exist
    for file in [model_file, model_info_file]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file not found: {file}")

    # Load model and model info
    best_model = joblib.load(model_file)
    with open(model_info_file, "r") as f:
        model_info = f.read()

    return best_model, model_info


# Load model and components
try:
    best_model, model_info = load_files()
except Exception as e:
    print(f"Error loading model files: {e}")
    best_model, model_info = None, "Model loading failed."

# Define feature groups
numeric_columns = ['Age', 'Tumor size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']
form_options = {
    '6th Stage': ['IA', 'IB', 'IC', 'IIA', 'IIB', 'IIC', 'IIIA', 'IIIB', 'IIIC'],
    'N Stage': ['N1', 'N2', 'N3'],
    'T Stage': ['T1', 'T2', 'T3'],
    'Grade': ['Well differentiated; Grade I', 'Moderately differentiated; Grade II', 'Poorly differentiated; Grade III']
}

# Routes
@app.route('/', methods=['GET'])
def home():
    """
    Render the home page with dynamic form inputs.
    """
    return render_template(
        'index.html',
        form_options=form_options,
        numeric_columns=numeric_columns,
        model_info=model_info
    )


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle form submission and return prediction results.
    """
    if not best_model:
        return jsonify({'error': 'Model is not loaded properly.'}), 500

    try:
        # Collect and process form data
        data = request.form.to_dict()

        # Ensure the necessary fields are included in the form
        for col in numeric_columns:
            if col not in data:
                return jsonify({'error': f'Missing required numeric field: {col}'}), 400

        # Convert the form data into a DataFrame
        df = pd.DataFrame([data])

        # Convert numeric fields to numeric type
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle categorical fields
        for col, values in form_options.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 1 if x in values else 0)

        # Ensure that the DataFrame is not empty and has the correct shape
        if df.empty or df.shape[1] != len(numeric_columns) + len(form_options):
            return jsonify({'error': 'Invalid input data or missing columns'}), 400

        # Make predictions
        prediction = best_model.predict(df)
        prediction_proba = best_model.predict_proba(df)

        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'probability': round(float(prediction_proba[0][1]), 4),
            'model_type': model_info
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/test', methods=['POST'])
def test():
    """
    Test the model with new data provided in JSON format.
    """
    if not best_model:
        return jsonify({'error': 'Model is not loaded properly.'}), 500

    try:
        # Parse the JSON payload
        new_data = request.get_json()

        if not new_data:
            return jsonify({'error': 'No data provided. Please send a valid JSON payload.'}), 400

        # Convert the JSON payload into a DataFrame
        df_new_data = pd.DataFrame([new_data])

        # Check for missing required fields
        missing_fields = [col for col in numeric_columns if col not in df_new_data.columns]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400

        # Convert numeric fields to numeric type
        for col in numeric_columns:
            if col in df_new_data.columns:
                df_new_data[col] = pd.to_numeric(df_new_data[col], errors='coerce')

        # Check for invalid or missing values
        if df_new_data.isnull().values.any():
            return jsonify({'error': 'Invalid or missing values in input data.'}), 400

        # Make predictions using the loaded model
        prediction = best_model.predict(df_new_data)
        prediction_proba = best_model.predict_proba(df_new_data)

        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'probability': round(float(prediction_proba[0][1]), 4),
            'model_type': model_info
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error during testing: {e}")
        return jsonify({'error': str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True)
    


