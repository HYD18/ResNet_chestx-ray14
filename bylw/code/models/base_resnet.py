### 基础resnet模型
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


class BaseResNet:
    def __init__(self, model_name='resnet18', num_classes=14, pretrained=True):
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.pretrained = pretrained

    def build(self):
        # 选择不同深度的ResNet
        model_dict = {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152
        }

        if self.model_name not in model_dict:
            raise ValueError(f"Unsupported model: {self.model_name}")

        model = model_dict[self.model_name](pretrained=self.pretrained)

        # 修改最后一层全连接层（适配多标签分类）
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.num_classes)

        return model


# 测试代码
if __name__ == '__main__':
    model = BaseResNet('resnet18').build()
    print(model)  # 打印模型结构
    dummy_input = torch.randn(1, 3, 224, 224)  # 测试输入
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 应为 [1, 14]