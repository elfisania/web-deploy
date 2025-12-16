from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

print("Loading models...")

models = {}
model_files = {
    'wine_nb': 'models/wine_nb.joblib',
    'wine_id3': 'models/wine_id3.joblib',
    'cancer_nb': 'models/cancer_nb.joblib',
    'cancer_id3': 'models/cancer_id3.joblib'
}

for model_name, file_path in model_files.items():
    try:
        models[model_name] = joblib.load(file_path)
        print(f"✓ Loaded {model_name} from {file_path}")
    except Exception as e:
        print(f"✗ Error loading {model_name}: {e}")
        models[model_name] = None

print(f"Successfully loaded {sum(1 for m in models.values() if m is not None)}/4 models\n")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict/<dataset>/<model_type>', methods=['POST'])
def api_predict(dataset, model_type):
    try:
        data = request.get_json()
        features = data.get('features', [])
        
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        
        valid_datasets = ['wine', 'cancer']
        valid_models = ['nb', 'id3']
        
        if dataset not in valid_datasets:
            return jsonify({'error': f'Invalid dataset. Must be one of: {valid_datasets}'}), 400
        
        if model_type not in valid_models:
            return jsonify({'error': f'Invalid model type. Must be one of: {valid_models}'}), 400
        

        model_key = f"{dataset}_{model_type}"
        model = models.get(model_key)
        
        if model is None:
            return jsonify({'error': f'Model {model_key} not loaded'}), 404
        

        expected_features = 11 if dataset == 'wine' else 30
        if len(features) != expected_features:
            return jsonify({'error': f'Expected {expected_features} features, got {len(features)}'}), 400
        

        prediction = model.predict([features])[0]

        if dataset == 'cancer':

            if prediction == 1:
                diagnosis = 'Malignant (Ganas)'
            else:
                diagnosis = 'Benign (Jinak)'

            result = {
                'prediction': str(prediction),  
                'type': 'diagnosis',
                'label': diagnosis,
                'message': f'Diagnosis: {diagnosis}'
            }
            return jsonify(result)

        else:
            result = {
                'prediction': int(prediction),
                'type': 'quality',
                'message': f'Wine Quality: {int(prediction)}'
            }
            return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


@app.route('/predict_wine', methods=['POST'])
def predict_wine():
    try:
        algorithm = request.form.get('algorithm', 'nb')
        model_key = 'wine_nb' if algorithm == 'nb' else 'wine_id3'
        model = models.get(model_key)
        
        if model is None:
            return render_template('wine.html', 
                                   error=f"Model {model_key} not loaded")
        
        features = [
            float(request.form['fixed_acidity']),
            float(request.form['volatile_acidity']),
            float(request.form['citric_acid']),
            float(request.form['residual_sugar']),
            float(request.form['chlorides']),
            float(request.form['free_sulfur_dioxide']),
            float(request.form['total_sulfur_dioxide']),
            float(request.form['density']),
            float(request.form['pH']),
            float(request.form['sulphates']),
            float(request.form['alcohol'])
        ]
        
        prediction = model.predict([features])[0]
        algo_name = "Naive Bayes" if algorithm == 'nb' else "Decision Tree (ID3)"
        
        return render_template('wine.html', 
                               prediction=int(prediction),
                               algorithm=algo_name,
                               features=features)
    
    except Exception as e:
        return render_template('wine.html', error=f"Prediction error: {str(e)}")


@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    try:
        algorithm = request.form.get('algorithm', 'nb')
        model_key = 'cancer_nb' if algorithm == 'nb' else 'cancer_id3'
        model = models.get(model_key)
        
        if model is None:
            return render_template('cancer.html', 
                                   error=f"Model {model_key} not loaded")
        
        features = []
        feature_names = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'smoothness_mean', 'compactness_mean', 'concavity_mean',
            'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se',
            'smoothness_se', 'compactness_se', 'concavity_se',
            'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
            'smoothness_worst', 'compactness_worst', 'concavity_worst',
            'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]
        
        for feature_name in feature_names:
            value = request.form[feature_name].strip()
            value = value.replace(',', '.')
            value = ''.join(c for c in value if c.isdigit() or c in '.-')

            try:
                features.append(float(value))
            except ValueError:
                return render_template('cancer.html', 
                    error=f"Format angka tidak valid untuk {feature_name}: {value}")
        
        prediction = model.predict([features])[0]

        if prediction == 1:
            result_text = "Malignant (Ganas)"
        else:
            result_text = "Benign (Jinak)"
        
        algo_name = "Naive Bayes" if algorithm == 'nb' else "Decision Tree (ID3)"
        
        return render_template('cancer.html', 
                               prediction=result_text,
                               prediction_value=str(prediction),
                               algorithm=algo_name,
                               features=features)
    
    except Exception as e:
        return render_template('cancer.html', 
                               error=f"Prediction error: {str(e)}")


@app.route('/result')
def result():
    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
