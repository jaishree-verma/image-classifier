import os
import sys
import torch
import requests
from flask import Flask, render_template, request, redirect, url_for
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.resnet import load_model
from utils.preprocess import preprocess_image

app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")


model = load_model()

# Load ImageNet labels once
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(labels_url).text.splitlines()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    filename = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = os.path.join("static", file.filename)
            file.save(filename)

            input_tensor = preprocess_image(filename)
            with torch.no_grad():
                output = model(input_tensor)

            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            pred_class = probabilities.argmax().item()

            prediction = labels[pred_class]
            confidence = probabilities[pred_class].item() * 100

    return render_template("index.html", prediction=prediction, confidence=confidence, filename=filename)

if __name__ == "__main__":
    app.run(debug=True)
