import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from models.resnet34_cbam import ResNet34_CBAM  # 修改为你的模型路径
# from model.resnet import ResNet34           # 如果是原始ResNet34，取消注释这行
import torch.nn.functional as F

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Grad-CAM 类 -----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # 归一化
        return cam

# ----------------- 图像预处理 -----------------
def preprocess_image(image_path):
    raw_img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(rgb_img).unsqueeze(0).to(device)
    return input_tensor, rgb_img

# ----------------- 可视化叠加 -----------------
def show_cam_on_image(rgb_img, mask, save_path=None):
    rgb_img = cv2.resize(rgb_img, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + np.float32(rgb_img) / 255
    overlay = overlay / np.max(overlay)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title("Grad-CAM")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# ----------------- 主函数入口 -----------------
if __name__ == "__main__":
    # === 修改为你保存模型的路径 ===
    model_path = "/root/autodl-tmp/project/result/resnet34_cbam_best.pth"
    image_path = "/root/autodl-tmp/project/bylw/result/test1.png"  # 替换为你的图像路径
    class_index = 0  # 设置你要可视化的类别编号（0~13）

    # === 加载模型 ===
    model = ResNet34_CBAM()  # 或 ResNet34()，根据你的需要
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # === 图像处理 & Grad-CAM 生成 ===
    input_tensor, raw_img = preprocess_image(image_path)
    grad_cam = GradCAM(model, model.layer4[-1])
    cam_mask = grad_cam(input_tensor, class_idx=class_index)

    # === 叠加可视化 & 保存 ===
    save_path = "/root/autodl-tmp/project/bylw/result/grad_cam_result.png"
    show_cam_on_image(raw_img, cam_mask, save_path)
    print(f"Grad-CAM 可视化已保存到 {save_path}")
