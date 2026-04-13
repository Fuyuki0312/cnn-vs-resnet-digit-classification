from torchvision.models import resnet18
from torch import nn

def Model_detecting_number():
    model = resnet18(weights=None)
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=10 # There are 10 subjects (0-9) to be classified
    )

    return model