import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from torch import optim
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder


"""
不用 GPU 训练将很慢
"""
def run():
    # 超参数
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    validation_split = 0.2  # 验证集比例

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # VGG16 需要 224x224 的输入
        transforms.Lambda(lambda img: img.convert('RGB')),  # 转换 RGBA 为 RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    data_dir = '../dataset/pics'  # 替换为你的数据集路径
    full_dataset = ImageFolder(root=data_dir, transform=transform)

    # 划分数据集
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


    # 加载 VGG16 模型
    model = models.vgg16(pretrained=False)  # 加载预训练模型
    model.classifier[6] = nn.Linear(4096, len(full_dataset.classes))  # 修改最后一层
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        model.train()  # 进入训练模式
        for images, labels in tqdm(train_loader):
            images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 验证模型
        model.eval()  # 进入评估模式
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Validation Accuracy of the model: {100 * correct / total:.2f}%')


if __name__ == "__main__":
    run()