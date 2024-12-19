import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets,id,transform=None):
        self.data = np.array(data/255).astype(np.float32).transpose(0, 3, 1, 2)
        self.targets = targets
        self.id = id
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        id = self.id[index]

        img =  Image.fromarray((x[0] * 255).astype(np.uint8))
        if self.transform is not None:
            x = self.transform(img)
        return x, y,id
def split_and_normalize_as_tensors(X, y, test_size=0.25, seed=0):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    return X_train, X_test, y_train, y_test


def train_val_split_tensors(X, y, val_size=0.25, seed=0):
    torch.manual_seed(seed)
    perm = torch.randperm(len(y))
    train_idx = perm[:int(val_size * len(y))]
    val_idx = perm[int(val_size * len(y)):]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def subsample(X, y, n=1000):
    mask = np.random.permutation(len(y))[:n]
    return X[mask], y[mask]



