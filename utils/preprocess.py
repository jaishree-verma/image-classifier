from torchvision.models import ResNet50_Weights
from PIL import Image

def preprocess_image(img_path):
    """
    Preprocesses an image for ResNet-50 using official ResNet50 weights transforms.
    """
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    image = Image.open(img_path).convert("RGB")
    return preprocess(image).unsqueeze(0)  # Add batch dimension
