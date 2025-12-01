# ResNet_chestx-ray14
基于深度学习的肺部疾病分类<br/>
本文目的在于研究基于深度学习构建一个高效的肺部疾病分类系统，提高胸部CT图像的自动分类正确率。<br/>
以ResNet34为基础模型，引入CBAM（Convolutional Block Attention Module）注意力机制，增强模型对病灶区域的关注能力。
<br/>
通过数据清洗、增强和标签处理等方法来优化输入数据，使用多标签分类机制和二元交叉熵损失函数对模型进行训练，结合Adam优化器和动态学习率调整策略提升性能。
<br/>
实验结果表明，改进后的ResNet34+CBAM模型在测试集上的宏平均F1分数达到0.1217，AUC值为0.8197，优于传统ResNet系列模型。<br/>
Grad-CAM可视化模型可以有效聚焦病灶区域，验证其可解释性。<br/>
本研究提出的融合注意力机制的深度学习模型能够显著提升肺部疾病分类的准确性和鲁棒性，为计算机辅助诊断系统提供了可行的解决方案。
<br/>
本文采用ChestX-ray14数据集，里边有112,120张胸部X光图像和14类肺部疾病标签。<br/><br/><br/>
数据集下载可在bylw文件夹内的README.md文件中找到<br/>
<br/>
核心代码均存放在code文件夹内，可以直接在pycharm中打开bylw文件夹，以便运行整个模型。<br/>
<br/>
各部分代码介绍：<br/>
models：存放核心模型代码<br/>
 &emsp;base_resnet：基础resnet模型(本文研究尚未使用)<br/>
 &emsp;cbam：CBAM注意力机制代码实现部分<br/>
 &emsp;resnet：包含resnet18、34、50、101、152<br/>
 &emsp;resnet34_cbam：在resnet34的基础上引入cbam注意力机制<br/>
<br/>
utils：工具包文件夹<br/>
 &emsp;cheset_dataset：数据集预处理代码<br/>
 &emsp;plot_results:结果绘制代码<br/>
<br/>
mian.py: 数据集预处理测试<br/>
train.py：模型训练文件<br/>
train_cbam.py：引入注意力机制的模型训练文件<br/>
grad_cam.py：grad-cam可视化<br/>
predict_auc_compare.py：绘制比较各AUC结果<br/>
<br/>
结果图如下：<br/>
![myplot8](https://github.com/user-attachments/assets/cd7d090e-2a36-4422-8737-56cb68786a2b)
![myplot7](https://github.com/user-attachments/assets/a7f6f283-54be-4473-8066-8521deec97d2)
<br/>
![cam]
<img width="1057" height="353" alt="image" src="https://github.com/user-attachments/assets/38e3b8a2-b1e7-43ba-9c41-054356897d21" />




