from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('waste_classifier_vgg16.h5')# Your saved model
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class_names = ['Biodegradable', 'Recyclable', 'Trash']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded!', 400

    img_file = request.files['image']
    path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
    img_file.save(path)

    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return render_template('index.html', prediction=predicted_class, image_file=img_file.filename)

if __name__ == '__main__':
    app.run(debug=True)
