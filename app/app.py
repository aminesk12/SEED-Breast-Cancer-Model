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
    categorical_columns = os.path.join(model_dir, " categorical_columns.pkl")
    feature_order = os.path.join(model_dir, "feature_order.pkl")

    # Ensure necessary files exist
    for file in [model_file, model_info_file]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file not found: {file}")

    # Load model and model info
    best_model = joblib.load(model_file)
    with open(model_info_file, "r") as f:
        model_info = f.read()

    return best_model,columns_order,categorical_columns, model_info

# Load model and components
try:
    best_model, columns_order, categorical_columns, model_info = load_files()
except Exception as e:
    print(f"Error loading model files: {e}")
    best_model, model_info = None, "Model loading failed."

# Define feature groups
numeric_columns = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']
form_options = {
    '6th Stage': ['IA', 'IB', 'IC', 'IIA', 'IIB', 'IIC', 'IIIA', 'IIIB', 'IIIC'],
    'N Stage': ['N1', 'N2', 'N3'],
    'T Stage': ['T1', 'T2', 'T3'],
    'Grade': ['Well differentiated; Grade I', 'Moderately differentiated; Grade II', 'Poorly differentiated; Grade III']
}

# Define column order for predictions
columns_order = ['Age', '6th Stage', 'Tumor Size', 'Grade', 'T Stage', 'N Stage',
       'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']

@app.route('/', methods=['GET'])
def home():
    return render_template(
        'index.html',
        form_options=form_options,
        numeric_columns=numeric_columns,
        model_info=model_info
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()

        # Create DataFrame with a single row
        df = pd.DataFrame([data])

        # Convert numeric fields
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])

        # Apply label encoding to categorical columns
        for col in categorical_columns:
            if col in df.columns:
                # Assuming label_encoders are loaded and available for encoding
                df[col] = label_encoders[col].transform(df[col])

        # Ensure the columns match the expected order
        df = df[columns_order]

     
        # Make prediction
        prediction = best_model.predict(scaled_features)
        prediction_proba = best_model.predict_proba(scaled_features)

        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'probability': float(prediction_proba[0][1]),
            'model_type': model_info
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Log the full error
        return jsonify({'error': str(e)}), 400  
    

if __name__ == '__main__':
    app.run(debug=True)
