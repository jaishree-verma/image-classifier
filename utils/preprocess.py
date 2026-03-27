from torchvision import transforms
from PIL import Image

def preprocess_image(img_path):
    """
    Preprocesses an image for ResNet-50:
    - Resize to 256
    - Center crop to 224
    - Convert to tensor
    - Normalize with ImageNet mean/std
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension
