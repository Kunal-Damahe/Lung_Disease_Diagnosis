from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        num_features = self.model.fc.in_features

        # New classifier
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2)
        )

        # Train only final layer
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)