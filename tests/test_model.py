import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import get_model

"""
Just checking the model actually builds and produces the right output shape
for a CIFAR-10-sized batch. Nothing fancy.
"""


def test_get_model_output_shape():
    model = get_model(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_get_model_num_classes_configurable():
    model = get_model(num_classes=5)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    assert out.shape == (1, 5)
