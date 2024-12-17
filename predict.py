import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models

#モデルを作る
model = models.MyModel()
print(model)

