import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def mnist(shuffle=True, batch_size=64):
    
    train_images = []
    train_labels = []
    for i in range(5):
        data = np.load(f"../../../data/corruptmnist/train_{i}.npz")
        [train_images.append(img) for img in data["images"]]
        [train_labels.append(label) for label in data["labels"]]

    train_images = np.array(train_images)
    # reshape to  (n_immgs, channels, pixels, pixels)
    train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
    train_labels = np.array(train_labels)

    train = [
        torch.from_numpy(train_images).type(torch.float32),
        torch.from_numpy(train_labels).long(),
    ]

    test_images = []
    test_labels = []
    data = np.load("../../../data/corruptmnist/test.npz")
    [test_images.append(img) for img in data["images"]]
    [test_labels.append(label) for label in data["labels"]]

    test_images = np.array(test_images)
    test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)
    test_labels = np.array(test_labels)
    test = [
        torch.tensor(test_images).type(torch.float32),
        torch.tensor(test_labels).long(),
    ]

    train_dataset = TensorDataset(train[0], train[1])  # create your datset
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )  # create your dataloader

    test_dataset = TensorDataset(test[0], test[1])  # create your datset
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size
    )  # create your dataloader

    return train_loader, test_loader
