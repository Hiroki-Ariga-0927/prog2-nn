import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32,scale=True)
])

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform  

)

ds_test = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform  

)

batch_size = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size,
)

for image_batch, label_batch in dataloader_test:
    print(image_batch.shape)
    print(label_batch.shape)
    break

def test_accuracy(model, dataloader):

    n_corrects = 0

    model.eval()
    for image_batch, label_batch in dataloader:

        with torch.no_grad():
            logits_batch = model(image_batch)

        predict_batch = logits_batch.argmax(dim=1)
        n_corrects += (label_batch == predict_batch).sum().item()

    accuracy = n_corrects / len(dataloader.dataset)

    return accuracy

