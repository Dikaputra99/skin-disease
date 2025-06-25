from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import os
import joblib
import json
from werkzeug.utils import secure_filename
import colorsys

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Load models and metadata
try:
    # Load metadata
    with open(os.path.join(MODEL_FOLDER, 'model_metadata.json')) as f:
        metadata = json.load(f)
    
    # Initialize models
    models = {}
    penyakit_list = metadata['classes']
    feature_order = metadata['feature_order']
    
    # Load each model
    for penyakit in penyakit_list:
        model_path = os.path.join(MODEL_FOLDER, f'model_{penyakit}.pkl')
        models[penyakit] = joblib.load(model_path)
        
    print("✅ Models loaded successfully:")
    print(f"Loaded {len(penyakit_list)} models: {', '.join(penyakit_list)}")
    print(f"Feature order: {feature_order}")

except Exception as e:
    print(f"❌ Error loading models: {str(e)}")
    print("Please run latih_knn_model.py first to train and save the models")
    exit(1)

def prepare_features(input_data):
    """
    Prepare features in the exact same way as during training
    Supports both dictionary (from JSON) and array (from image) inputs
    """
    if isinstance(input_data, dict):
        r, g, b = input_data['R'], input_data['G'], input_data['B']
    else:
        r, g, b = input_data
    
    # Convert to 0-1 range
    r_norm, g_norm, b_norm = r/255, g/255, b/255
    
    # Calculate all features (same as in latih_knn_model.py)
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    features = {
        'R': r,
        'G': g,
        'B': b,
        'H': h,
        'S': s,
        'V': v,
        'Brightness': (r + g + b) / 3,
        'RG_Ratio': r / (g + 1e-6),
        'RB_Ratio': r / (b + 1e-6),
        'GB_Ratio': g / (b + 1e-6),
        'Luminance': 0.299*r + 0.587*g + 0.114*b
    }
    
    # Return features in the exact same order as during training
    return [features[col] for col in feature_order]

@app.route("/")
def home():
    return {
        "message": "Skin Disease Classification API",
        "available_diseases": penyakit_list,
        "expected_features": feature_order
    }

@app.route("/get_diseases", methods=["GET"])
def get_diseases():
    return jsonify({
        "diseases": penyakit_list,
        "count": len(penyakit_list)
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # Validate input
        if not all(k in data for k in ["R", "G", "B"]):
            return jsonify({
                "error": "Incomplete RGB data",
                "required": ["R", "G", "B"],
                "received": list(data.keys())
            }), 400
        
        # Prepare features
        input_features = prepare_features(data)
        
        # Make predictions
        results = {}
        for penyakit in penyakit_list:
            proba = models[penyakit].predict_proba([input_features])[0][1]
            results[penyakit] = {
                "prediction": "Positive" if proba >= 0.5 else "Negative",
                "confidence": round(proba * 100, 2),
                "threshold": 0.5
            }
        
        return jsonify({
            "status": "success",
            "predictions": results,
            "feature_values": dict(zip(feature_order, input_features))
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/predict_image", methods=["POST"])
def predict_image():
    try:
        # Validate file
        if "file" not in request.files:
            return jsonify({
                "error": "No file uploaded",
                "required": "file (image)"
            }), 400
        
        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Save and process image
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            image = Image.open(filepath).convert("RGB")
            np_image = np.array(image)
            
            # Calculate average RGB
            r_avg = np.mean(np_image[:, :, 0])
            g_avg = np.mean(np_image[:, :, 1])
            b_avg = np.mean(np_image[:, :, 2])
            
            # Prepare features
            input_features = prepare_features([r_avg, g_avg, b_avg])
            
            # Make predictions
            results = {}
            for penyakit in penyakit_list:
                proba = models[penyakit].predict_proba([input_features])[0][1]
                results[penyakit] = {
                    "prediction": "Positive" if proba >= 0.5 else "Negative",
                    "confidence": round(proba * 100, 2),
                    "threshold": 0.5
                }
            
            return jsonify({
                "status": "success",
                "predictions": results,
                "average_rgb": {
                    "R": round(r_avg),
                    "G": round(g_avg),
                    "B": round(b_avg)
                },
                "feature_values": dict(zip(feature_order, input_features))
            })
            
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)