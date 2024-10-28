import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from PIL import Image

app = Flask(__name__)

# Load the models
model_keras = tf.keras.models.load_model('models/cnn_cifar10_model.keras')
model_h5 = tf.keras.models.load_model('models/cnn_cifar10_model.h5')

# CIFAR-10 class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Image preprocessing function
def prepare_image(image):
    image = image.resize((32, 32))  # CIFAR-10 images are 32x32 pixels
    image = np.array(image).astype('float32') / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file uploaded", 400
    
    file = request.files['file']
    image = Image.open(file.stream)
    prepared_image = prepare_image(image)

    # Make predictions with both models
    keras_pred = class_names[np.argmax(model_keras.predict(prepared_image))]
    h5_pred = class_names[np.argmax(model_h5.predict(prepared_image))]

    # Render the results in result.html
    return render_template('result.html', keras_result=keras_pred, h5_result=h5_pred)

if __name__ == '__main__':
    app.run(debug=True)
