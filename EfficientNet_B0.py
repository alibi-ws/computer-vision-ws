import torch
import torch.nn as nn


class SEBlock(nn.Module):
  def __init__(self, in_channels, reduction=16):
    super().__init__()
    self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.fc1 = nn.Linear(in_features=in_channels, out_features=in_channels//reduction)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(in_features=in_channels//reduction, out_features=in_channels)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    out = self.gap(x)
    out = torch.flatten(out, 1)
    out = self.fc1(out)
    out = self.relu(out)
    out = self.fc2(out)
    out = self.sigmoid(out)
    out = out.reshape(x.size(0), x.size(1), 1, 1)
    out = x * out

    return out


class MBBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, exp_factor=6, 
               stride=1, se_reduction=16):
    super().__init__()
    self.use_residual = (in_channels == out_channels) and (stride == 1)
    midd_channels = in_channels * exp_factor
    self.expansion = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=midd_channels,
                  kernel_size=1, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(midd_channels),
        nn.SiLU(inplace=True)
    ) if exp_factor != 1 else nn.Identity()

    self.depthwise = nn.Sequential(
        nn.Conv2d(in_channels=midd_channels, out_channels=midd_channels, 
                  kernel_size=kernel_size, stride=stride, padding=1, 
                  groups=midd_channels, bias=False),
        nn.BatchNorm2d(midd_channels),
        nn.SiLU(inplace=True)
    )

    self.se = SEBlock(in_channels=midd_channels, reduction=se_reduction)

    self.project = nn.Sequential(
        nn.Conv2d(in_channels=midd_channels, out_channels=out_channels, 
                  kernel_size=1, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True)
    )

  def forward(self, x):
    out = self.expansion(x)
    out = self.depthwise(out)
    out = self.se(out)
    out = self.project(out)
    if self.use_residual:
      out += x
  
    return out


class EfficientNetB0(nn.Module):
  def __init__(self, num_classes):
    super().__init__()

    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, 
                  padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.SiLU(inplace=True)
    )
    self.last_out_channels = 32
    self.layer0 = self._make_layer(MBBlock, 1, 16, 3, 1, 1)
    self.layer1 = self._make_layer(MBBlock, 2, 24, 3, 6, 2)
    self.layer2 = self._make_layer(MBBlock, 2, 40, 5, 6, 2)
    self.layer3 = self._make_layer(MBBlock, 3, 80, 3, 6, 2)
    self.layer4 = self._make_layer(MBBlock, 3, 112, 5, 6, 1)
    self.layer5 = self._make_layer(MBBlock, 4, 192, 5, 6, 2)
    self.layer6 = self._make_layer(MBBlock, 1, 320, 3, 6, 1)
    self.conv2 = nn.Sequential(
        nn.Conv2d(self.last_out_channels, 1280, 1, bias=False),
        nn.BatchNorm2d(1280),
        nn.SiLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(1280, num_classes)
    )

  def _make_layer(self, block, num_blocks, out_channels, 
                  kernel_size, exp_factor, stride, se_reduction=16):
    layers = []
    layers.append(block(self.last_out_channels, out_channels, kernel_size, 
                        exp_factor, stride, se_reduction))
    for i in range(1, num_blocks):
      layers.append(block(out_channels, out_channels, kernel_size, 
                        exp_factor, 1, se_reduction))
    self.last_out_channels = out_channels

    return(nn.Sequential(*layers))

  def forward(self, x):
    out = self.conv1(x)
    out = self.layer0(out)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = self.layer6(out)
    out = self.conv2(out)

    return out