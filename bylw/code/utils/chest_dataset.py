import os
from collections import Counter

import torch
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms
from bylw.code.utils.transforms import data_transforms  # 本地导入

# 数据加载  创建完整的数据集实例
class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        # 读取CSV时确保处理路径正确
        self.labels_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir  # 直接使用相对路径
        self.transform = transform

        # ---------------- 数据清洗 ----------------
        # 处理缺失值（如果标签为空，默认设为"无标签"）
        self.labels_frame['Finding Labels'].fillna('No Finding', inplace=True)

        # 验证前5个文件是否存在
        for i in range(5):
            img_name = self.labels_frame.iloc[i, 0]
            img_path = os.path.join(self.image_dir, img_name)
            if not os.path.exists(img_path):
                print(f"警告: 文件不存在 ({i}): {img_path}")

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        # 在子目录中查找图像
        img_name = self.labels_frame.iloc[idx, 0]
        img_path = self._find_image_in_subfolders(img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[ERROR] 图像文件不存在: {img_path}")

        image = Image.open(img_path).convert('RGB')

        # ----------------应用数据增强 ----------------
        if self.transform:
            image = self.transform(image)

        # 读取标签（第2列）
        label_text = self.labels_frame.iloc[idx, 1]
        labels = label_text.split('|')
        target = [1 if cls in labels else 0 for cls in self.get_classes()]
        target = torch.tensor(target, dtype=torch.float)

        return image, target

    def _find_image_in_subfolders(self, img_name):
        for root, _, files in os.walk(self.image_dir):
            if img_name in files:
                return os.path.join(root, img_name)
        return os.path.join(self.image_dir, img_name)  # fallback

    def get_classes(self):
        return [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

# 测试代码
if __name__ == "__main__":
    # 确保工作目录是项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
    os.chdir(project_root)
    print("切换到项目根目录:", project_root)

    # 数据路径
    image_dir = os.path.join("bylw", "data", "images")  # 图像文件夹路径
    full_csv = os.path.join("bylw", "data", "Data_Entry_2017_v2020.csv")  # CSV 文件路径

    # 定义保存路径
    save_dir = os.path.join("bylw", "data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_csv = os.path.join(save_dir, 'train.csv')  # 训练集 CSV 文件路径
    val_csv = os.path.join(save_dir, 'val.csv')  # 验证集 CSV 文件路径

    # 创建完整的数据集实例
    full_dataset = ChestXrayDataset(
        csv_file=full_csv,
        image_dir=image_dir,
        transform=None  # 初始不应用transform
    )

    # ----------------动态划分训练集和验证集 ----------------
    train_size = int(0.8 * len(full_dataset))  # 80% 用于训练  训练集
    val_size = len(full_dataset) - train_size  # 剩余20% 用于验证  测试集
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 设置不同的transform
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    # 打印数据集大小
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 查看一个样本
    image, target = train_dataset[0]
    print(f"Sample image shape: {image.shape}")
    print(f"Sample target: {target}")

    # 查看一个样本
    image, target = train_dataset[0]
    print(f"Sample image shape: {image.shape}")
    print(f"Sample target: {target}")

    # 验证标签分布
    all_labels = []
    for i in range(len(full_dataset)):
        label_text = full_dataset.labels_frame.iloc[i, 1]
        labels = label_text.split('|')
        all_labels.extend(labels)

    label_counts = Counter(all_labels)
    print("Label distribution:", label_counts)

    # 检查图像和标签的对应关系
    def visualize_sample(dataset, index):
        image, target = dataset[index]
        print("Target:", target)
        plt.imshow(image.permute(1, 2, 0))  # 将 Tensor 转换为图像格式
        plt.show()
    visualize_sample(train_dataset, 0)
    visualize_sample(val_dataset, 0)

    # ---------------- 保存训练集和验证集 ----------------
    # 获取划分后的索引
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices

    # 根据索引提取对应的记录
    train_df = full_dataset.labels_frame.iloc[train_indices].reset_index(drop=True)
    val_df = full_dataset.labels_frame.iloc[val_indices].reset_index(drop=True)

    # 保存为 CSV 文件
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print(f"训练集已保存到: {train_csv}")
    print(f"验证集已保存到: {val_csv}")