import torch
import torch.nn as nn
import torchvision.models as models

class ResNetMultiLabel(nn.Module):
    def __init__(self, base_model='resnet50', num_classes=14, pretrained=True):
        super(ResNetMultiLabel, self).__init__()

        # 选择模型
        if base_model == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif base_model == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
        elif base_model == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif base_model == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        elif base_model == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError("Unsupported base_model: " + base_model)

        in_features = self.backbone.fc.in_features  # 获取原始全连接层的输入特征数

        # 替换分类层：输出14维多标签
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Sigmoid()  # 多标签输出概率
        )

    def forward(self, x):
        return self.backbone(x)
