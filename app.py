from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf

# derive directories
app_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(app_dir, 'model', 'saved_model')
model_path = os.path.join(model_dir, 'mnist_cnn.h5')

# import after deriving paths
from model.utils import get_prediction, save_example

app = Flask(__name__)
model = None

def load_model():
    global model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run model/train.py first.")
        return False
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None and not load_model():
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    img = data.get('image')
    if not img:
        return jsonify({'error': 'No image provided'}), 400

    try:
        result = get_prediction(model, img)
        if data.get('save', False):
            actual = data.get('actual_digit')
            save_example(img, result['digit'], actual)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    img = data.get('image')
    pred = data.get('predicted')
    actual = data.get('actual')
    if img is None or pred is None or actual is None:
        return jsonify({'error': 'Missing fields'}), 400

    try:
        fname = save_example(img, pred, actual)
        return jsonify({'success': True, 'file': fname})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True)