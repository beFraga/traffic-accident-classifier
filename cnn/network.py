import torch
import torch.nn as nn
import torchvision.models as models

class AccidentClassifierNet(nn.Module):
    def __init__(self, num_classes=5, S=7):
        super(AccidentClassifierNet, self).__init__()

        self.S = S
        self.num_classes = num_classes
    
        channels = [3, 16, 32, 64, 128, 128]
        self.backbone = nn.Sequential(*[
            self._conv_block(channels[i], channels[i+1]) for i in range(len(channels) - 1)
        ])

        out_channels = 1 + self.num_classes + 4

        self.head = nn.Sequential(
            nn.Conv2d(channels[-1], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


    def forward(self, x):
        x = self.backbone(x)
        out = self.head(x)

        out = out.permute(0,2,3,1)

        return out
    


class TransferResnet(nn.Module):
    def __init__(self, num_classes=5, S=7):
        super(TransferResnet, self).__init__()

        self.S = S
        self.num_classes = num_classes

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze the early/generic layers (conv1, bn1, relu, maxpool, layer1, layer2)
        # to reduce trainable capacity and preserve pretrained low-level features,
        # which curbs overfitting on a small dataset. Later layers stay trainable.
        for param in self.backbone[:6].parameters():
            param.requires_grad = False

        out_channels = 1 + self.num_classes + 4

        self.head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.3),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.3),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )


    def forward(self, x):
        x = self.backbone(x)
        out = self.head(x)

        out = out.permute(0,2,3,1)

        return out