import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    expansion_factor = 1

    def __init__(self, in_ch, out_ch, stride_val=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.convolution1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                      stride=stride_val, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_ch)
        self.activation = nn.ReLU(inplace=True)
        self.convolution2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                                      stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_ch)
        self.shortcut = shortcut

    # y = F(x) + x

    def forward(self, x):
        out = self.activation(self.batchnorm1(self.convolution1(x)))
        out = self.batchnorm2(self.convolution2(out))
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        out = self.activation(out)
        return out


class CustomResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet18, self).__init__()
        self.current_channels = 64

        self.initial_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial_bn = nn.BatchNorm2d(64)
        self.activation = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._build_stage(64, 2, stride=1)
        self.stage2 = self._build_stage(128, 2, stride=2)
        self.stage3 = self._build_stage(256, 2, stride=2)
        self.stage4 = self._build_stage(512, 2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _build_stage(self, out_channels, num_blocks, stride):
        shortcut = None
        if stride != 1 or self.current_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(self.current_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [ResidualBlock(self.current_channels, out_channels, stride, shortcut)]
        self.current_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pooling(self.activation(self.initial_bn(self.initial_conv(x))))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
