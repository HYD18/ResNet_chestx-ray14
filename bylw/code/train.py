import os
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from torchvision import transforms
from tqdm import tqdm
from utils.transforms import data_transforms
from bylw.code.utils.chest_dataset import ChestXrayDataset
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

# ======================= 固定随机种子 =======================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 加速卷积

set_seed()

# ======================= 全局设置 =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备：", device)
num_classes = 14
batch_size = 128  # 增大 batch size
num_epochs = 20
model_names = ['resnet18']
# model_names = ['resnet34', 'resnet50', 'resnet101', 'resnet152']   # 先跑34 50 101 152

# 获取根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(BASE_DIR, 'bylw', 'data')
image_dir = os.path.join(data_dir, 'images')
train_csv = os.path.join(data_dir, 'train.csv')
val_csv = os.path.join(data_dir, 'val.csv')
save_dir = os.path.join(BASE_DIR, 'result')
os.makedirs(save_dir, exist_ok=True)

# ======================= 数据加载器 =======================
def create_dataloaders():
    train_dataset = ChestXrayDataset(train_csv, image_dir, data_transforms['train'])
    val_dataset = ChestXrayDataset(val_csv, image_dir, data_transforms['val'])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, prefetch_factor=4
    )
    return train_loader, val_loader

# ======================= 模型工厂 =======================
def build_model(model_name):
    return {
        'resnet18': ResNet18(),
        'resnet34': ResNet34(),
        'resnet50': ResNet50(),
        'resnet101': ResNet101(),
        'resnet152': ResNet152()
    }[model_name]

# ======================= 保存每轮指标 =======================
def save_epoch_results(epoch, model_name, train_loss, val_loss, val_f1, val_auc):
    path = os.path.join(save_dir, f"{model_name}_epoch_{epoch + 1}.csv")
    pd.DataFrame({
        'epoch': [epoch + 1],
        'train_loss': [train_loss],
        'val_loss': [val_loss],
        'val_f1': [val_f1],
        'val_auc': [val_auc]
    }).to_csv(path, index=False)
    print(f"已保存 Epoch {epoch + 1} 指标到: {path}")

# ======================= 训练函数 =======================
def train_model(model, train_loader, val_loader, model_name):
    if torch.cuda.device_count() > 1:
        print(f"使用多GPU训练：{torch.cuda.device_count()} 个 GPU")
        model = torch.nn.DataParallel(model)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    best_val_loss = float('inf')
    results = {'model': model_name, 'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_auc': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        val_f1 = f1_score(all_targets, (all_preds > 0.5).astype(int), average='macro')
        val_auc = roc_auc_score(all_targets, all_preds, average='macro')

        # 保存模型和日志
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_best.pth"))
            print(f"\n 最佳模型已保存: {model_name} | Val Loss: {val_loss:.4f}")

        print(f"\n[Epoch {epoch + 1}/{num_epochs}] {model_name}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")

        save_epoch_results(epoch, model_name, train_loss, val_loss, val_f1, val_auc)
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)
        results['val_f1'].append(val_f1)
        results['val_auc'].append(val_auc)

    # 保存整体训练日志
    pd.DataFrame(results).to_csv(os.path.join(save_dir, f"{model_name}_log.csv"), index=False)
    return results

# ======================= 主函数 =======================
def main():
    train_loader, val_loader = create_dataloaders()
    all_results = []
    for model_name in model_names:
        print(f"\n 开始训练模型：{model_name}")
        model = build_model(model_name)
        results = train_model(model, train_loader, val_loader, model_name)
        all_results.append(results)

    summary = pd.DataFrame({
        'model': [r['model'] for r in all_results],
        'best_val_loss': [min(r['val_loss']) for r in all_results],
        'best_val_f1': [max(r['val_f1']) for r in all_results],
        'best_val_auc': [max(r['val_auc']) for r in all_results]
    })
    summary.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)
    print("\n 所有模型训练完成，结果保存至 model_comparison.csv")

if __name__ == '__main__':
    main()