import torchvision
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, feat_dim=1280, output_dim=4):
        super(Model, self).__init__()

        self.feat_dim = feat_dim
        self.output_dim = output_dim

        # MobileNetV2
        self.m = torchvision.models.mobilenet_v2(pretrained=True)  # feat_dim=1280

        # Adjust the last classifier layer to the number of our classes
        self.m.classifier = nn.Linear(feat_dim, output_dim)

    def forward(self, img):
        out = self.m(img)
        return out
