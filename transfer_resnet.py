import kagglehub
from torchvision import transforms, datasets, models
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch
import torch.optim as optim
import gc

# Download latest version of dataset form kaggle
path = kagglehub.dataset_download("emmarex/plantdisease")

print("Path to dataset files:", path)

def data_loader(data_dir, test_size=0.15, val_size=0.15, batch_size=64):
  # Load the full dataset
  full_dataset = datasets.ImageFolder(root=data_dir)

  full_indices = list(range(len(full_dataset)))
  full_targets = full_dataset.targets

  # Stratified split based on indices and targets
  train_val_indices, test_indices = train_test_split(full_indices,
                                                      test_size=test_size,
                                                      random_state=42,
                                                      stratify=full_targets)

  train_val_targets = [full_targets[i] for i in train_val_indices]
  val_size = val_size / (100 - test_size)
  train_indices, val_indices = train_test_split(train_val_indices,
                                                test_size=val_size,
                                                random_state=42,
                                                stratify=train_val_targets)

  # Seperate transformers for train and test, train have data augmentation
  train_transformer = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(15),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
  ])

  val_test_transformer = transforms.Compose([
      transforms.Resize(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
  ])

  # Create seperate datasets for train, val, and test
  train_dataset = Subset(datasets.ImageFolder(root=data_dir,
                                              transform=train_transformer),
                         indices=train_indices)
  val_dataset = Subset(datasets.ImageFolder(root=data_dir,
                                            transform=val_test_transformer),
                       indices=val_indices)
  test_dataset = Subset(datasets.ImageFolder(root=data_dir,
                                             transform=val_test_transformer),
                        indices=test_indices)

  # Create data loaders
  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                            shuffle=True)
  val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                          shuffle=False)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                           shuffle=False)

  num_classes = len(full_dataset.classes)

  return train_loader, val_loader, test_loader, num_classes


train_loader, val_loader, test_loader, num_classes = data_loader(f'{path}/PlantVillage')

def train_model(model, criterion, optimizer, train_loader, val_loader,
                num_epochs, device):
  for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')

    epoch_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
      images = images.to(device)
      labels = labels.to(device)

      # Clear gradients for each batch
      optimizer.zero_grad()
      # Calculate model's output for each batch
      outputs = model(images)
      # Calculate loss
      loss = criterion(outputs, labels)
      # Calculate gradients based on loss and parameters
      loss.backward()
      # Update parameters based on the gradients
      optimizer.step()

      epoch_loss += loss.item()

      # # Clear gpu and cache
      # del images, labels, outputs
      # torch.cuda.empty_cache()
      # gc.collect()
      print('\r', '*'*(i//5), '-'*((len(train_loader)-i)//5), end='')

    print(f'Loss: {epoch_loss/len(train_loader):.4f}')

    val_loss = 0.0
    # Disable gradient calculation for validation phase
    with torch.no_grad():
      correct = 0
      total = 0

      for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        # Return the maximu value in class arrays per input, returns the
        # predicted class by mximum probability
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (labels == pred).sum().item()

        # del images, labels, outputs
      print(f'Validation Loss is {val_loss/len(val_loader):.4f} and Validation Accuray is {100*correct/total:.2f}%')
      print('-' * 10)
  return model


def test_model(model, test_loader, device):
  # Disable gradient calculation for test phase
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)

      # Outputs the predcted class by choosing highest probability between
      # classes per input
      _, pred = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (labels == pred).sum().item()

      # del images, labels, outputs

    print(f'Test accuracy is {100*correct/total:.2f}%')

# Call resnet50 pre-trained model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Freez model's backbone
for param in model.parameters():
  param.requires_grad = False

# Replace model's classification layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Choose loss
criterion = nn.CrossEntropyLoss()
# Choose optimizer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Train model's fc layer only
model = train_model(model, criterion, optimizer, train_loader, val_loader, 5,
                    device)
test_model(model, test_loader, device)

# Unfreez model's last 10 layers for fine-tuning
for param in model.layer4.parameters():
  param.requires_grad = True

# Choose optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)
# Train model's last 10 layers for fine-tuning
model = train_model(model, criterion, optimizer, train_loader, val_loader, 3,
                    device)
test_model(model, test_loader, device)

# Save the model
torch.save(model.state_dict(), 'plan_disease_resnet50.pth')

