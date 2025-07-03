import torch.nn as nn
from torchvision import models

class MobileNetV2FER(nn.Module):
    def __init__(self, num_classes=7):
        super(MobileNetV2FER, self).__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Modifica a primeira camada convolucional para aceitar 1 canal (escala de cinza)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Modifica a camada classificadora final
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, num_classes)
        )

    def forward(self, x):
        return self.model(x)


