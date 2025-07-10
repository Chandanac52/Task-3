from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load model and scaler
with open("../model/iris_model.pkl", "rb") as f:
    scaler, model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return "🌸 Iris Classifier API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return jsonify({
            "prediction": int(prediction[0]),
            "class_name": ["setosa", "versicolor", "virginica"][int(prediction[0])]
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
