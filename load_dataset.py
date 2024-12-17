import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
)

print(f'num of database: {len(ds_train)}')

image, target = ds_train[1]
print(type(image),target)

plt.imshow(image,cmap='gray_r')
plt.title(target)
plt.show

image = transforms.functional.to_image(image)
print(image.shape, image.dtype)
print(image.minimum(), image.max())
