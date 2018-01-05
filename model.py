import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import numpy as np
import cv2


# Label: [0->5]
DATA_DIR = 'data'
LABELS = open(os.path.join(DATA_DIR, 'labels.txt'), 'r').read().split('\n')[:-1]


def predict(model, X):
    """Arguments
        X: numpy image, shape (H, W, C)
    """
    # gray
    X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
    # rescale
    X = cv2.resize(X, (64, 64))
    # normalize
    X = X / 255.
    # reshape
    X = np.reshape(X, (1, 1, 64, 64))

    X = torch.from_numpy(X.astype('float32'))
    X = Variable(X, volatile=True)

    # forward
    output = model.cpu()(X)

    # get index of the max
    _, index = output.data.max(1, keepdim=True)
    return LABELS[index[0][0]] # index is a LongTensor -> need to get int data


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32*13*13, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, len(LABELS)),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) # flatten [m, C, H, W] -> [m, C*H*W]
        out = self.fc1(out)
        out = self.fc2(out)
        return out
