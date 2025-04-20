import torch
import torch.nn as nn
import torch.nn.functional as F

'''-------------一、BasicBlock模块-----------------------------'''


class BasicBlock(nn.Module):
    expansion = 1  # 用于标记BasicBlock的通道扩展倍数（BasicBlock无扩展）

    def __init__(self, inchannel, outchannel, stride=1):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


'''-------------二、Bottleneck模块-----------------------------'''


class Bottleneck(nn.Module):
    expansion = 4  # Bottleneck的通道扩展倍数为4

    def __init__(self, inchannel, outchannel, stride=1):
        super(Bottleneck, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel * self.expansion, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outchannel * self.expansion),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel * self.expansion)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


'''-------------三、通用ResNet类-----------------------------'''


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=14):  # 修改为14类（根据你的数据集）
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 标准ResNet的7x7卷积
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化，适配不同输入尺寸
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, outchannel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, outchannel, stride))
            self.inchannel = outchannel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


'''-------------四、ResNet工厂函数-----------------------------'''


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=14)  # 14类
def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=14)  # 14类
def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=14)  # 14类
def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=14)  # 14类
def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=14)  # 14类