from torch import nn

import src.models.ecg_resnet


class ModelWrapper(nn.Module):
    def __init__(self, arch, num_classes):
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes

        self.backbone = getattr(src.models.ecg_resnet, arch)()
        self.classifier = nn.Linear(self.backbone.rep_dim, self.num_classes)

    def forward(self, x):
        rep = self.backbone(x)
        cls = self.classifier(rep)
        return cls
