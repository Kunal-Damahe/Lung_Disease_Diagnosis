from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, pretrained=True):
        super(Net, self).__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        num_features = self.model.fc.in_features

        # Fine-tune only the highest-level residual features. Lower layers
        # remain frozen to retain useful ImageNet edge/texture detectors.
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # New classifier
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 2)
        )

        # Train only final layer
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module in self.model.modules():
                if isinstance(module, nn.BatchNorm2d) and all(
                    not parameter.requires_grad for parameter in module.parameters()
                ):
                    module.eval()
        return self
