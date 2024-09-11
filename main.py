import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from torch import optim
from tqdm import tqdm


class CWRUDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def load_dataset(batch_size=32):
    data = np.load("./dataset/CWRU_48k_load_1_CNN_data.npz")
    data_array = data['data']
    labels_array = data['labels']
    # print(labels_array.shape)
    # print(data_array.shape)
    # print(labels_array[0:10])
    unique_labels = np.unique(labels_array)
    # print("Unique labels:", unique_labels)
    data_array = data_array.reshape(data_array.shape[0], 1, data_array.shape[1]*data_array.shape[2])
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    indexed_labels = [label_to_index[label] for label in labels_array]

    # 转换为 PyTorch 张量
    indexed_tensor = torch.tensor(indexed_labels)
    # 进行 one-hot 编码
    num_classes = len(unique_labels)
    labels_torch = torch.nn.functional.one_hot(indexed_tensor, num_classes=num_classes).to(torch.float32)
    data_torch = torch.tensor(data_array, dtype=torch.float32)
    dataset = CWRUDataset(data_torch, labels_torch)
    # 切分数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(train_dataset), len(val_dataset)


class CWRUModel_01(nn.Module):
    def __init__(self, conv1d_sizes=[64,32,16,8,4], kernel_size=3, class_num=10, feature_size=1024):
        super(CWRUModel_01, self).__init__()
        self.conv1ds = []
        for idx in conv1d_sizes:
            self.conv1ds.append(
                [
                    nn.Conv1d(in_channels=1, out_channels=idx, kernel_size=kernel_size, padding=1),
                    nn.Conv1d(in_channels=1, out_channels=idx, kernel_size=kernel_size, padding=1)
                ]
            )
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, feature_size))
        self.linner = nn.Linear(feature_size, class_num)
        
    def forward(self, x):
        result = []
        for convid_parts in self.conv1ds:
            x_01 = convid_parts[0](x)            # 32 x channel x 1024
            x_01 = self.relu(x_01)               # 32 x channel x 1024
            x_01 = self.global_avg_pool(x_01)    # 32 x 1 x 1024
            x_01 = convid_parts[0](x_01)         # 32 x channel x 1024
            x_01 = self.relu(x_01)               # 32 x channel x 1024
            x_01 = self.global_avg_pool(x_01)    # 32 x 1 x 1024
            result.append(x_01)

        x = torch.sum(torch.stack(result), dim=0) # 32 x 1 x 1024
        x = x.squeeze(1)                          # 32 x 1024
        # 全连接层
        x = self.linner(x)                        # 32 x 10
        x = torch.softmax(x, dim=1)               # 32 x 10

        return x


def run():
    train_loader, eval_loader, len_of_train, len_of_eval = load_dataset(batch_size=32)
    print(len_of_train, len_of_eval)
    model = CWRUModel_01(feature_size=1024)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    i = 0

    for epoch in range(2):
        for batch_data, batch_labels in tqdm(train_loader):
            out = model(batch_data)
            # print(batch_labels.shape)
            loss = criterion(batch_labels, out)
            optimizer.zero_grad()
            loss.backward()
            # 更新权重
            optimizer.step()
            i += 1
            if i % 20 == 0:
                print("loss = {}".format(loss.item()))
    
    model.eval()

    eval_loss = []
    eval_correct = 0

    with torch.no_grad():
        for batch_data, batch_labels in tqdm(eval_loader):
            out = model(batch_data)
            loss = criterion(out, batch_labels)
            eval_loss.append(loss.item())
            _, pred = out.max(1)
            _, labels = batch_labels.max(1)
            num_correct = (pred == labels).sum().item()
            # print(num_correct)
            eval_correct += num_correct
    print("len(test_loader) = {}".format(len(eval_loader)))
    # 求平均
    loss = np.array(eval_loss)
    avg_loss = np.mean(loss)
    print(avg_loss)
    print("avg loss = {}, eval_acc = {}".format(avg_loss, eval_correct/len_of_eval))


if __name__ == "__main__":
    run()