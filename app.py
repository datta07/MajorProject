from __future__ import division, print_function

import os

import numpy as np

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask libraries
# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Datta_Yoga_Pose.h5'

# Load your trained model
model = load_model(MODEL_PATH)

reverse_mapping = {
    0: 'virabhadrasana i',
    1: 'padmasana',
    2: 'savasana',
    3: 'vajrasana',
    4: 'vrischikasana',
    5: 'vriksasana',
    6: 'utthita hasta padangustasana',
    7: 'virabhadrasana ii',
    8: 'ustrasana',
    9: 'dandasana',
    10: 'bakasana',
    11: 'tittibhasana',
    12: 'tolasana',
    13: 'salamba sarvangasana',
    14: 'prasarita padottanasana',
    15: 'uttanasana',
    16: 'tadasana',
    17: 'parivrtta janu sirsasana',
    18: 'parivrtta trikonasana',
    19: 'paripurna navasana',
    20: 'lolasana',
    21: 'ardha pincha mayurasana',
    22: 'simhasana',
    23: 'natarajasana',
    24: 'hanumanasana',
    25: 'durvasasana'
}


def mapper(value):
    return reverse_mapping[value]


def final_predicts(image_path, model):
    img = image.load_img(image_path, target_size=(40, 40))

    # Preprocessing the image
    x = image.img_to_array(img)

    # Scaling

    x = x / 255
    prediction_image = np.expand_dims(x, axis=0)

    prediction = model.predict(prediction_image)
    value = np.argmax(prediction)
    move_name = mapper(value)
    return move_name


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        submission_result = final_predicts(file_path, model)

        return submission_result

    return None


if __name__ == '__main__':
    app.run(debug=True)
