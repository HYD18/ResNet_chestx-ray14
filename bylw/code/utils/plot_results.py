# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator, FuncFormatter  # 导入刻度格式化工具

# 设置字体
font = FontProperties(fname='/root/autodl-tmp/project/SimHei.ttf')
mpl.rcParams['axes.unicode_minus'] = False

print("字体名称:", font.get_name())


def plot_training_log(csv_path, model_name):
    df = pd.read_csv(csv_path)

    if 'epoch' not in df.columns:
        df.insert(0, 'epoch', range(1, len(df) + 1))  # 自动补 epoch，从1开始计数

    epochs = df['epoch']
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{model_name} Training Metrics', fontsize=16)

    # 绘制 Loss 曲线
    axs[0, 0].plot(epochs, df['train_loss'], label='Train Loss', color='blue')
    axs[0, 0].plot(epochs, df['val_loss'], label='Val Loss', color='orange')
    axs[0, 0].set_title('Loss Curve', fontproperties=font)
    axs[0, 0].legend(prop=font)

    # 绘制 F1 Score 曲线
    if 'val_f1' in df.columns:
        axs[0, 1].plot(epochs, df['val_f1'], label='Val F1', color='green')
        axs[0, 1].set_title('F1 Score', fontproperties=font)
    else:
        axs[0, 1].axis('off')  # 隐藏空白图

    # 绘制 AUC Score 曲线
    if 'val_auc' in df.columns:
        axs[1, 0].plot(epochs, df['val_auc'], label='Val AUC', color='purple')
        axs[1, 0].set_title('AUC Score', fontproperties=font)
    else:
        axs[1, 0].axis('off')  # 隐藏空白图

    # 遍历所有子图，设置 x 和 y 轴的样式
    for ax in axs.flat:
        ax.set_xlabel('训练轮数 (Epoch)', fontproperties=font)
        ax.set_ylabel('', fontproperties=font)
        ax.grid(True)

        # 优化 x 轴刻度：设置最大刻度数量为10个
        ax.xaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))

        # 格式化 x 轴刻度标签
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))  # 显示整数

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    save_path = os.path.join(os.path.dirname(csv_path), f"{model_name}_metrics_plot.png")
    plt.savefig(save_path)
    plt.show()


# 多模型AUC对比图
def plot_auc_comparison(log_dir, model_list):
    plt.figure(figsize=(10, 6))

    for csv_file, label in model_list:
        df = pd.read_csv(os.path.join(log_dir, csv_file))
        if 'epoch' not in df.columns:
            df.insert(0, 'epoch', range(1, len(df) + 1))  # 自动补 epoch，从1开始计数

        plt.plot(df['epoch'], df['val_auc'], label=label)

    plt.title("模型对比：验证AUC变化曲线", fontsize=14, fontproperties=font)
    plt.xlabel("训练轮数 (Epoch)", fontproperties=font)
    plt.ylabel("AUC 分数", fontproperties=font)
    plt.legend(prop=font)
    plt.grid(True)

    # 优化 x 轴刻度：设置最大刻度数量为10个
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))

    # 格式化 x 轴刻度标签
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))  # 显示整数

    plt.savefig(os.path.join(log_dir, 'val_auc_comparison.png'))
    plt.show()


if __name__ == '__main__':
    # 获取当前文件的绝对路径（这段代码放在 bylw/code/utils/plot_results.py 中）
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

    # 修改 log_dir 为实际路径
    log_dir = "/root/autodl-tmp/project/result"

    model_list = [
        ('resnet18_log.csv', 'ResNet18'),
        ('resnet34_log.csv', 'ResNet34'),
        ('resnet50_log.csv', 'ResNet50'),
        ('resnet101_log.csv', 'ResNet101'),
        ('resnet152_log.csv', 'ResNet152'),
        #  添加注意力机制模型的结果文件
        ('resnet34_cbam_log.csv', 'ResNet34 + CBAM'),
    ]

    # 绘制每个模型的训练曲线
    for csv_file, model_name in model_list:
        csv_path = os.path.join(log_dir, csv_file)
        plot_training_log(csv_path, model_name)

    # 绘制模型对比图
    plot_auc_comparison(log_dir, model_list)