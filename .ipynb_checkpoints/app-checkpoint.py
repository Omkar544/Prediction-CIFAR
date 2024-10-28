import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
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
    image = np.array(image)
    image = image.astype('float32') / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        # Prepare the image for prediction
        image = Image.open(file.stream)
        prepared_image = prepare_image(image)

        # Make predictions with the models
        predictions_keras = model_keras.predict(prepared_image)
        predicted_class_keras = class_names[np.argmax(predictions_keras)]

        predictions_h5 = model_h5.predict(prepared_image)
        predicted_class_h5 = class_names[np.argmax(predictions_h5)]

        # Print the results for debugging
        print(f"Keras Prediction: {predicted_class_keras}")
        print(f"H5 Prediction: {predicted_class_h5}")

        # Render the result in result.html
        return render_template('result.html', 
                               keras_result=predicted_class_keras, 
                               h5_result=predicted_class_h5)

if __name__ == '__main__':
    app.run(debug=True)
