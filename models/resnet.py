from torchvision import models

def load_model():
    """
    Loads a pre-trained ResNet-50 model from torchvision.
    Sets it to evaluation mode for inference.
    """
    model = models.resnet50(pretrained=True)
    model.eval()
    return model
