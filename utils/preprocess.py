from torchvision.models import ResNet18_Weights
from PIL import Image

def preprocess_image(img_path):
    """
    Preprocesses an image for ResNet-18 using official ResNet18 weights transforms.
    """
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    image = Image.open(img_path).convert("RGB")
    return preprocess(image).unsqueeze(0)  # Add batch dimension
