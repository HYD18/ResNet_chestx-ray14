import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from bylw.code.utils.chest_dataset import ChestXrayDataset
from bylw.code.utils.transforms import data_transforms
from models.resnet import ResNet34
from models.resnet34_cbam import ResNet34_CBAM  # 已实现了CBAM模块

# 设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 14
batch_size = 128
save_dir = "/root/autodl-tmp/project/result"
data_dir = "/root/autodl-tmp/project/bylw/data"
image_dir = os.path.join(data_dir, "images")
val_csv = os.path.join(data_dir, "val.csv")

# 加载验证集
val_dataset = ChestXrayDataset(val_csv, image_dir, data_transforms['val'])
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# 获取标签名
class_names = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]



def evaluate_model(model, model_name):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_targets.append(targets.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # 每类AUC
    per_class_auc = []
    for i in range(num_classes):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            print(f"⚠️ 类别 {i} 无法计算 AUC（标签全为0或全为1），将填充为 NaN")
            auc = np.nan
        per_class_auc.append(auc)

    # 打印长度确认
    print(f"[DEBUG] 类别总数: {len(class_names)}")
    print(f"[DEBUG] AUC 数量: {len(per_class_auc)}")

    # 构建 DataFrame 时安全切片（避免长度不一致）
    min_len = min(len(class_names), len(per_class_auc))
    auc_df = pd.DataFrame({
        'Class': class_names[:min_len],
        model_name: per_class_auc[:min_len]
    })

    auc_df.to_csv(os.path.join(save_dir, f"{model_name}_per_class_auc.csv"), index=False)
    return auc_df



def load_model(model_class, weight_path):
    model = model_class()
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    return model


# 加载并评估两个模型
resnet34 = load_model(ResNet34, os.path.join(save_dir, "resnet34_best.pth"))
resnet34_cbam = load_model(ResNet34_CBAM, os.path.join(save_dir, "resnet34_cbam_best.pth"))

df1 = evaluate_model(resnet34, "ResNet34")
df2 = evaluate_model(resnet34_cbam, "ResNet34+CBAM")

# 合并AUC结果
merged = pd.merge(df1, df2, on="Class")
merged.to_csv(os.path.join(save_dir, "ResNet34_vs_CBAM_AUC.csv"), index=False)

# 绘制柱状图
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(num_classes)
plt.bar(index, merged["ResNet34"], bar_width, label="ResNet34")
plt.bar(index + bar_width, merged["ResNet34+CBAM"], bar_width, label="ResNet34+CBAM")
plt.xlabel("Disease Class")
plt.ylabel("AUC")
plt.title("Per-Class AUC Comparison")
plt.xticks(index + bar_width / 2, merged["Class"], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "resnet34_vs_cbam_auc_barplot.png"))
plt.show()
