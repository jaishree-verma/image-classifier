from torchvision import models
from torchvision.models import ResNet50_Weights

def load_model():
    """
    Loads a pre-trained ResNet-50 model using the new weights API.
    """
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()
    return model