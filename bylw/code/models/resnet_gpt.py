import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from resnet_multi_label import ResNetMultiLabel
def train_one_model(model, train_loader, val_loader, model_name, epochs=10, lr=1e-4, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # 多标签分类用 BCE

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"✅ [Epoch {epoch+1}] 训练损失：{avg_loss:.4f}")

        # 你可以在这里加 val 验证逻辑
        # ...

    # 保存模型
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    torch.save(model.state_dict(), f"{model_name}_{timestamp}.pth")
    print(f"✅ 模型 {model_name} 已保存\n")

    # 启动多个模型训练
    model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

    for name in model_names:
        print(f"\n 正在训练模型：{name}")
        model = ResNetMultiLabel(base_model=name, num_classes=14, pretrained=True)
        train_one_model(model, train_loader, val_loader, model_name=name, epochs=5, device='cuda')

