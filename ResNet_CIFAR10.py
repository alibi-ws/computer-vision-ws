import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms
import numpy as np
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_loader(data_dir, batch_size, valid_size=0.1, random_seed=42,
                shuffle=True, test=False):
  # Defining the normalizer
  normalizer = transforms.Normalize(
    mean = [0.4914, 0.4822, 0.4465],
    std = [0.2023, 0.1994, 0.2010])
  # Defining the data transformer
  transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalizer])

  if test:
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transformer)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size)
    return test_loader

  # For train and validation
  train_dataset = datasets.CIFAR10(
      root=data_dir,
      download=True,
      train=True,
      transform=transformer)

  num_train = len(train_dataset)
  indices = list(range(num_train))
  split = int(np.floor(valid_size * num_train))

  if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

  train_idx, val_idx = indices[split:], indices[:split]
  train_sampler = SubsetRandomSampler(train_idx)
  val_sampler = SubsetRandomSampler(val_idx)

  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             sampler=train_sampler)
  val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           sampler=val_sampler)

  return (train_loader, val_loader)

train_loader, val_loader = data_loader('./data', batch_size=64)
test_loader = data_loader('./data', batch_size=64, test=True)

# Creating residual block
class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    # Super is needed to initialize nn.Module __init__ method
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU())
    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels))
    # If shape doesnt match, a downsampler is needed on the skip connection to match the shape of x to output
    if in_channels != out_channels or stride != 1:
      self.downsample = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
          nn.BatchNorm2d(out_channels)
      )
    self.relu = nn.ReLU()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stride = stride

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.conv2(out)
    if self.in_channels != self.out_channels or self.stride != 1:
      residual = self.downsample(x)
    out += residual
    out = self.relu(out)

    return out

# Creat ResNet
class ResNet(nn.Module):
  def __init__(self, block, layers):
    super().__init__()
    self.blocks_in_channels = 64
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=self.blocks_in_channels,
                  kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(self.blocks_in_channels),
        nn.ReLU()
    )
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer0 = self._make_layer(block, layers[0], 64, 1)
    self.layer1 = self._make_layer(block, layers[1], 128, 2)
    self.layer2 = self._make_layer(block, layers[2], 256, 2)
    self.layer3 = self._make_layer(block, layers[3], 512, 2)
    self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
    self.fc = nn.Linear(512, 10)

  def _make_layer(self, block, num_blocks, out_channels, stride):
    layers = []
    layers.append(block(self.blocks_in_channels, out_channels, stride))
    self.blocks_in_channels = out_channels
    for i in range(1, num_blocks):
      layers.append(block(self.blocks_in_channels, out_channels, 1))

    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.conv1(x)
    out = self.maxpool(out)
    out = self.layer0(out)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.avgpool(out)
    out = torch.flatten(out, 1)
    out = self.fc(out)

    return out

num_classes = 10
num_epochs = 20
batch_size = 16
learning_rate = 0.01

model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                            weight_decay = 0.001, momentum = 0.9)

for epoch in range (num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)
    # Calculate outputs
    outputs = model(images)
    # Calculate loss
    loss = criterion(outputs, labels)
    # Reset gradients for each batch
    optimizer.zero_grad()
    # Calculate gradients in backpropagation
    loss.backward()
    # Update model parameters based on calculated gradients
    optimizer.step()
    del images, labels, outputs
    torch.cuda.empty_cache()
    gc.collect()
  print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():4f}')

  # Set no_grad on validation to not occupy computation resources for gradients
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (labels == predicted).sum().item()
      del images, labels, outputs
    print(f"Validation accuracy on 5000 validation images is: {100*correct/total}%")

with torch.no_grad():
  correct = 0
  total = 0
  for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (labels == predicted).sum().item()
    del images, labels, outputs
  print(f"Test accuracy on 10000 test images is {100*correct/total}%")