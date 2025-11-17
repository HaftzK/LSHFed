import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import random
import numpy as np
import warnings
import os
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# model sourcecode used in experiment
# CIFAR ResNet9 model from https://github.com/Moddy2024/ResNet-9.git

warnings.filterwarnings("ignore")
DATA_AMOUNT = 5  # use 1/DATA_AMOUNT data

datasource = "MNIST"  # available: "CIFAR" "MNIST" "FMNIST"
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
if datasource == "MNIST":
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    subset_indices = random.sample(range(len(train_dataset)), len(train_dataset) // DATA_AMOUNT)
    train_subset = Subset(train_dataset, subset_indices)
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

elif datasource == "CIFAR":
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    subset_indices = random.sample(range(len(train_dataset)), len(train_dataset) // DATA_AMOUNT)
    train_subset = Subset(train_dataset, subset_indices)
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

elif datasource == "FMNIST":
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    subset_indices = random.sample(range(len(train_dataset)), len(train_dataset) // DATA_AMOUNT)
    train_subset = Subset(train_dataset, subset_indices)
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


if datasource == "MNIST":
    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()
            # 1 input channel (grayscale), 16 output channels, 3x3 convolution
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling
            self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 32 filters, 7x7 image size
            self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))  # Convolution -> ReLU -> MaxPooling
            x = self.pool(torch.relu(self.conv2(x)))  # Another conv -> ReLU -> MaxPooling
            x = x.view(-1, 32 * 7 * 7)  # Flatten the tensor for fully connected layer
            x = torch.relu(self.fc1(x))  # Fully connected layer with ReLU
            x = self.fc2(x)  # Output layer (no activation needed, we'll use softmax in the loss function)
            return x

# CIFAR ResNet9 model from https://github.com/Moddy2024/ResNet-9.git
elif datasource == "CIFAR":
    # ResNET9
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


    class ImageClassificationBase(nn.Module):
        def training_step(self, batch):
            images, labels = batch
            out = self(images)  # Generate predictions
            loss = F.cross_entropy(out, labels)  # Calculate loss
            acc = accuracy(out, labels)
            return loss, acc

        def validation_step(self, batch):
            images, labels = batch
            out = self(images)  # Generate predictions
            loss = F.cross_entropy(out, labels)  # Calculate loss
            acc = accuracy(out, labels)  # Calculate accuracy
            return {'val_loss': loss.detach(), 'val_acc': acc}

        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

        def epoch_end(self, epoch, result):
            print(
                "Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, last_lr: {:.5f}".format(
                    epoch + 1, result['train_loss'], result['train_accuracy'], result['val_loss'],
                    result['val_acc'], result['lrs'][-1]))


    def conv_block(in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        if pool: layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)


    class CustomModel(ImageClassificationBase):
        def __init__(self, in_channels, num_classes):
            super().__init__()

            self.conv1 = conv_block(in_channels, 64)
            self.conv2 = conv_block(64, 128, pool=True)
            self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

            self.conv3 = conv_block(128, 256, pool=True)
            self.conv4 = conv_block(256, 512, pool=True)
            self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

            self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                            nn.Flatten(),
                                            nn.Dropout(0.2),
                                            nn.Linear(512, num_classes))

        def forward(self, xb):
            out = self.conv1(xb)
            out = self.conv2(out)
            out = self.res1(out) + out
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out) + out
            out = self.classifier(out)
            return out

elif datasource == "FMNIST":
    BATCH_SIZE = 128
    EPOCHS = 10
    LR = 3e-4
    PATCH_SIZE = 7  # 7×7 → 4×4 = 16 patches per 28×28 image
    EMBED_DIM = 128
    NUM_LAYERS = 6
    NUM_HEADS = 8
    MLP_DIM = 256


    # DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    class CustomModel(nn.Module):
        def __init__(self, num_classes: int = 10):
            super().__init__()
            num_patches = (28 // PATCH_SIZE) ** 2

            # Patch embedding via conv (kernel = stride = patch)
            self.patch_embed = nn.Conv2d(1, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, EMBED_DIM))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=EMBED_DIM,
                nhead=NUM_HEADS,
                dim_feedforward=MLP_DIM,
                batch_first=True,
                dropout=0.1,
                activation="gelu"
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
            self.norm = nn.LayerNorm(EMBED_DIM)
            self.head = nn.Linear(EMBED_DIM, num_classes)

            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        def forward(self, x):
            B = x.size(0)
            x = self.patch_embed(x)  # (B, C, H', W')
            x = x.flatten(2).transpose(1, 2)  # (B, N, C)  — N = 16
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls, x), dim=1)  # (B, 17, C)
            x = x + self.pos_embed[:, : x.size(1), :]  # match length 17
            x = self.encoder(x)
            x = self.norm(x[:, 0])  # CLS token
            return self.head(x)

