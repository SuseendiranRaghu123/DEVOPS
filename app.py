import logging
from flask import Flask, request, jsonify, send_from_directory
import xgboost as xgb
import numpy as np
import joblib
import json
import os
from datetime import datetime

# -----------------------------
# Configure Logging to stdout
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("adhd_api")

# -----------------------------
# Flask Setup
# -----------------------------
app = Flask(
    __name__,
    static_folder="simon_says",    # your WebGL build folder
    static_url_path=""             # serve at root
)

# Load model once
model = joblib.load("adhd_models.pkl")

@app.route('/')
def serve_game():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    logger.info(f"[/predict] Received data: {data!r}")  # <-- log incoming JSON
    if not data or 'features' not in data:
        logger.warning("[/predict] Missing 'features' key")
        return jsonify({"error": "Invalid input. 'features' key missing."}), 400
    try:
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)[0]
        logger.info(f"[/predict] Prediction result: {prediction}")
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        logger.exception("[/predict] Exception during prediction")
        return jsonify({"error": str(e)}), 500

@app.route('/collect', methods=['POST'])
def collect_data():
    data = request.get_json()
    logger.info(f"[/collect] Received data: {data!r}")  # <-- log incoming JSON
    if not data:
        logger.warning("[/collect] No JSON data received")
        return jsonify({"error": "No JSON data received."}), 400
    try:
        os.makedirs("data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/game_data_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"[/collect] Data written to {filename}")
        return jsonify({"status": "success", "timestamp": timestamp}), 200
    except Exception as e:
        logger.exception("[/collect] Exception while saving data")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure Flask's own logger uses our handler level
    app.logger.setLevel(logging.INFO)
    app.run(host='0.0.0.0', port=5000)
