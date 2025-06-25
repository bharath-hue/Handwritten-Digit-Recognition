import os
import io
import re
import base64
import numpy as np
from PIL import Image, ImageOps

# derive directories
this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(this_dir)
samples_dir = os.path.join(project_root, 'collected_samples')

def preprocess_image(image_data):
    """
    Preprocess the image data from a canvas drawing for model prediction.
    """
    # strip header and decode
    image_data = re.sub(r'^data:image/.+;base64,', '', image_data)
    raw = base64.b64decode(image_data)
    
    # open & convert
    img = Image.open(io.BytesIO(raw)).convert('L').resize((28,28))
    img = ImageOps.invert(img)
    
    arr = np.array(img, dtype='float32') / 255.0
    return arr.reshape(1,28,28,1)

def get_prediction(model, image_data):
    """
    Return a dict with predicted digit, confidence, all_scores.
    """
    processed = preprocess_image(image_data)
    preds = model.predict(processed)[0]
    digit = int(np.argmax(preds))
    return {
        'digit': digit,
        'confidence': float(preds[digit]),
        'all_scores': {str(i): float(preds[i]) for i in range(10)}
    }

def save_example(image_data, predicted_digit, actual_digit=None):
    """
    Save an example canvas image under project_root/collected_samples.
    Returns the full filepath.
    """
    os.makedirs(samples_dir, exist_ok=True)

    # decode back to PIL image
    imgdata = re.sub(r'^data:image/.+;base64,', '', image_data)
    raw = base64.b64decode(imgdata)
    img = Image.open(io.BytesIO(raw))
    
    # build filename
    rnd = np.random.randint(1e4)
    if actual_digit is not None:
        fname = f"digit_{actual_digit}_pred_{predicted_digit}_{rnd}.png"
    else:
        fname = f"pred_{predicted_digit}_{rnd}.png"
    
    fullpath = os.path.join(samples_dir, fname)
    img.save(fullpath)
    return fullpath