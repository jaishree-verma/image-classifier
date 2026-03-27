import torch
import requests
import argparse
from models.resnet import load_model
from utils.preprocess import preprocess_image

def main(img_path):
    # Load pre-trained ResNet model
    model = load_model()
    # Preprocess image
    input_tensor = preprocess_image(img_path)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)

    # Convert to probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    pred_class = probabilities.argmax().item()

    # Load ImageNet labels
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = requests.get(labels_url).text.splitlines()

    # Print prediction
    print(f"Predicted: {labels[pred_class]} ({probabilities[pred_class].item()*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    main(args.image)
