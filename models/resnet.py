from torchvision import models
from torchvision.models import ResNet18_Weights

def load_model():
    """
    Loads a pre-trained, lightweight ResNet-18 model optimized for low-memory environments (<512MB RAM).
    """
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    return model