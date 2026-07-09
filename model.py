import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# === MODEL CONFIGURATION ===
DATASET = 'cifar10'         # 'mnist' | 'fashion_mnist' | 'cifar10' | 'cifar100' | 'emnist_balanced' | 'emnist_byclass'
MODEL_NAME = 'resnet18'     # 'net_mnist' | 'resnet18' | 'resnet34' | 'resnet50' | 'resnet101' | 'resnet152' | 'vgg18'


# === utils: map dataset -> (num_classes, input_channels) ===
def _dataset_spec(dataset: str):
    ds = dataset.lower()
    if ds == 'mnist':
        return 10, 1
    if ds == 'fashion_mnist':
        return 10, 1
    if ds == 'emnist_balanced' or ds == 'emnist-balanced':
        return 47, 1
    if ds == 'emnist_byclass' or ds == 'emnist-byclass' or ds == 'emnist-by-class':
        return 62, 1
    if ds == 'svhn':
        return 10, 3
    if ds == 'cifar10':
        return 10, 3
    if ds == 'cifar100':
        return 100, 3
    raise ValueError(f"Unsupported dataset '{dataset}'")


# === MODEL SELECTION INTERFACE ===
def model_init(config):
    dataset = config["dataset"]
    model_name = config["model_name"]

    num_classes, input_channels = _dataset_spec(dataset)

    # 灰度系数据集（MNIST/EMNIST/FMNIST）
    if dataset.lower() in ['mnist', 'fashion_mnist', 'emnist_balanced', 'emnist-balanced',
                           'emnist_byclass', 'emnist-byclass', 'emnist-by-class']:
        if model_name == 'net_mnist':
            return Net_mnist(num_classes=num_classes)
        elif model_name.startswith('resnet'):
            return load_resnet(model_name, input_channels=input_channels, num_classes=num_classes)
        elif model_name == 'vgg18':
            return load_vgg18(input_channels=input_channels, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model '{model_name}' for dataset '{dataset}'")

    # RGB 数据集（CIFAR10/100）
    elif dataset.lower() in ['cifar10', 'cifar100', 'svhn']:
        if model_name == 'vgg18':
            return load_vgg18(input_channels=input_channels, num_classes=num_classes)
        elif model_name.startswith('resnet'):
            return load_resnet(model_name, input_channels=input_channels, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model '{model_name}' for dataset '{dataset}'")

    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")


# === NET_MNIST ===
class Net_mnist(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# === LOAD RESNET (TORCHVISION) ===
def load_resnet(name, input_channels=3, num_classes=10):
    if name == 'resnet18':
        model = models.resnet18(weights=None, num_classes=num_classes)
    elif name == 'resnet34':
        model = models.resnet34(weights=None, num_classes=num_classes)
    elif name == 'resnet50':
        model = models.resnet50(weights=None, num_classes=num_classes)
    elif name == 'resnet101':
        model = models.resnet101(weights=None, num_classes=num_classes)
    elif name == 'resnet152':
        model = models.resnet152(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported ResNet variant: {name}")

    # CIFAR/MNIST/EMNIST-friendly: 小卷积核 + 去掉 maxpool
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# === LOAD VGG18 ===
def load_vgg18(input_channels=3, num_classes=10):
    model = models.vgg18(weights=None)
    if input_channels != 3:
        features = list(model.features)
        features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        model.features = nn.Sequential(*features)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model
