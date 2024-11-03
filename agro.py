from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(_name_)
model = load_model("cauliflower_disease_classifier.h5")
categories = ["Bacterial_spot_rot", "Black_rot", "Downy_mildew", "No_disease"]

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))  # Match the input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array / 255.0

UPLOAD_FOLDER = './uimag'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["image"]   
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    file_path =os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess and predict
    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    result = categories[class_index]

    os.remove(file_path)  # Clean up the saved file after prediction
    return jsonify({"classification": result})

if _name_ == "_main_":
    app.run(debug=True)