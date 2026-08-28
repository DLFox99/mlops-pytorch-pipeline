"""ResNet-18 for CIFAR-10.

Stock torchvision resnet18 is built for 224x224 ImageNet images. CIFAR-10
images are 32x32, so if you use the default stem (7x7 stride-2 conv +
maxpool) you lose most of the spatial resolution before the residual
blocks even get a chance to do anything. Swapping in a 3x3 stride-1 stem
and dropping the initial maxpool is the standard fix people use for
CIFAR-scale ResNets.
"""

import torch.nn as nn
from torchvision.models import resnet18


def get_model(num_classes: int = 10) -> nn.Module:
    """Build a ResNet-18 adapted for 32x32 inputs."""
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
