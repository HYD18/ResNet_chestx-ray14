import os

# 定义数据增强与数据加载器

from torch.utils.data import DataLoader, random_split
from bylw.code.utils.chest_dataset import ChestXrayDataset
import torchvision.transforms as transforms


# 加载数据集
def main():
    # 设置路径
    # 硬编码绝对路径（确保与您的实际路径一致）
    csv_path = r"F:\chestX-ray14\bylw\data\Data_Entry_2017_v2020.csv"   # 数据标签路径
    img_dir = r"F:\chestX-ray14\bylw\data\images"    # 数据集路径

    # 图像增强与标准化
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 加载数据集
    full_dataset = ChestXrayDataset(csv_file=csv_path, image_dir=img_dir, transform=train_transform)

    # 划分数据集
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # 加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print("图像文件总数：", len(os.listdir("F:/chestX-ray14/bylw/data/images")))
    total_images = count_images_in_subfolders(img_dir)
    print("图像总数为：", total_images)
    for images, labels in train_loader:
        print("图像 shape：", images.shape)
        print("标签示例：", labels[0])
        break

# 统计文件总数的类
def count_images_in_subfolders(root_dir, exts=('.png', '.jpg', '.jpeg')):
    total = 0
    for root, _, files in os.walk(root_dir):
        count = sum(1 for file in files if file.lower().endswith(exts))
        total += count
    return total


if __name__ == '__main__':
    main()