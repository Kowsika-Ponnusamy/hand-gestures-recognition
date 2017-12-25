import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np
from PIL import Image


# Label: [0->5]
CLASSES = range(6)


def predict(model, X):
    """Arguments
        X: numpy image, shape (H, W, C)
    """
    # resize to (64x64x3)
    X = Image.fromarray(X)
    transform = transforms.Resize(64)
    X = transform(X)

    # reshape
    X = np.asarray(X)
    shape = X.shape
    X = X.reshape((1, shape[2], shape[0], shape[1]))
    # normalize
    X = X / 255.

    X = torch.from_numpy(X.astype('float32'))
    X = Variable(X, volatile=True)

    # forward
    output = model.cpu()(X)

    # get index of the max
    _, index = output.data.max(1, keepdim=True)
    return CLASSES[index[0][0]] # index is a LongTensor -> need to get int data


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
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
            nn.ReLU()
        )
        self.fc2 = nn.Linear(1024, len(CLASSES))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) # flatten [m, C, H, W] -> [m, C*H*W]
        out = self.fc1(out)
        out = self.fc2(out)
        return out
