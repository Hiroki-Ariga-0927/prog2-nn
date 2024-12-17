import matplotlib.pyplot as plt
from torchvision import datasets

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