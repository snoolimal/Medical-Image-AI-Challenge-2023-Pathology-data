import torch
import torch.nn as nn
import timm


class ResNet(nn.Module):
    def __init__(self, num_channels):
        super(ResNet, self).__init__()

        self.pre_feature_extractor = nn.Sequential(
            nn.Conv2d(kernel_size=5, in_channels=3, out_channels=num_channels),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(kernel_size=5, in_channels=num_channels, out_channels=3)
        )

        self.backbone = timm.create_model('resnet101', pretrained=True)
        self.backbone.global_pool = nn.Conv2d(kernel_size=1, in_channels=2048, out_channels=512)
        self.backbone.fc = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=4096, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )

    def forward(self, x):
        x = self.pre_feature_extractor(x)
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        x = torch.sigmoid(x)

        return x