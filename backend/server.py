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

# Load ImageNet labels with fallback
try:
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    resp = requests.get(labels_url, timeout=5)
    if resp.status_code == 200:
        labels = resp.text.splitlines()
    else:
        from utils.labels import IMAGENET_LABELS
        labels = IMAGENET_LABELS
except Exception:
    from utils.labels import IMAGENET_LABELS
    labels = IMAGENET_LABELS

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    filename = None
    top5 = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename != "":
            static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "static"))
            os.makedirs(static_dir, exist_ok=True)
            save_path = os.path.join(static_dir, file.filename)
            file.save(save_path)
            filename = file.filename

            input_tensor = preprocess_image(save_path)
            with torch.no_grad():
                output = model(input_tensor)

            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            
            prediction = labels[top5_catid[0].item()]
            confidence = top5_prob[0].item() * 100
            
            top5 = [(labels[top5_catid[i].item()], top5_prob[i].item() * 100) for i in range(5)]

    return render_template("index.html", prediction=prediction, confidence=confidence, filename=filename, top5=top5)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
