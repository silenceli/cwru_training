from tsai.imports import *
from tsai.utils import *
from tsai.models.layers import *

from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim
from tqdm import tqdm
import torch

class _ResCNNBlock(Module):
    def __init__(self, ni, nf, kss=[7, 5, 3], coord=False, separable=False, zero_norm=False):
        self.convblock1 = ConvBlock(ni, nf, kss[0], coord=coord, separable=separable)
        self.convblock2 = ConvBlock(nf, nf, kss[1], coord=coord, separable=separable)
        self.convblock3 = ConvBlock(nf, nf, kss[2], act=None, coord=coord, separable=separable, zero_norm=zero_norm)

        # expand channels for the sum if necessary
        self.shortcut = ConvBN(ni, nf, 1, coord=coord)
        self.add = Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.add(x, self.shortcut(res))
        x = self.act(x)
        return x


class ResCNN(Module):
    def __init__(self, c_in, c_out, coord=False, separable=False, zero_norm=False):
        nf = 64
        self.block1 = _ResCNNBlock(c_in, nf, kss=[7, 5, 3], coord=coord, separable=separable, zero_norm=zero_norm)
        self.block2 = ConvBlock(nf, nf * 2, 3, coord=coord, separable=separable, act=nn.LeakyReLU, act_kwargs={'negative_slope':.2})
        self.block3 = ConvBlock(nf * 2, nf * 4, 3, coord=coord, separable=separable, act=nn.PReLU)
        self.block4 = ConvBlock(nf * 4, nf * 2, 3, coord=coord, separable=separable, act=nn.ELU, act_kwargs={'alpha':.3})
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)
        self.lin = nn.Linear(nf * 2, c_out)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.squeeze(self.gap(x))
        return self.lin(x)

xb = torch.rand(100, 32, 32)
model = ResCNN(32, 10, coord=True, separable=True)
#a = rescnn(xb)
#print(a.shape)

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


train_loader, eval_loader, len_of_train, len_of_eval = load_dataset(batch_size=32)
print(len_of_train, len_of_eval)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
i = 0

for epoch in range(40):
    for batch_data, batch_labels in tqdm(train_loader):
        out = model(batch_data)
        out = torch.softmax(out, dim=1)
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