import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

from dataset.CT_DataSet import MyCTDataSet
from model.model import ConvNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 150
batch_size = 64
num_classes = 2
dropout_prob = 0.3
weight_decay = 0.0003

# 加载模型
model = ConvNet(num_classes,dropout_prob).to(device)

# 加载数据
ds_train = MyCTDataSet("../dataset/train_images.npy", "../dataset/train_labels.npy")
train_loader = DataLoader(ds_train,batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

ds_val = MyCTDataSet("../dataset/val_images.npy", "../dataset/val_labels.npy")
val_loader = DataLoader(ds_val,batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),weight_decay=weight_decay)

# 训练
acc_best = 0
losses = {'train': [], 'val': []}  # 用于保存每个epoch的训练集和验证集损失值
for epoch in tqdm(range(epochs)):
    model.train()
    total_loss = 0
    total_tr = 0
    correct_tr = 0
    for iteration, (image, label) in enumerate(train_loader):
        image = image.to(torch.float).to(device)
        label = label.squeeze(-1).to(device)
        total_tr += label.size(0)
        optimizer.zero_grad()

        output=model(image)
        #print(output)
        loss=criterion(output,label)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

        pred = torch.argmax(output,dim=1)
        for i in range(batch_size):
            if pred[i]==label[i]:
                correct_tr+=1

    accuracy_tr = 100*correct_tr/total_tr
    average_loss_tr = total_loss / len(train_loader)

    # 模型验证
    model.eval()
    total_val = 0
    correct_val = 0
    total_loss_val = 0
    with torch.no_grad():
        for iteration, (image, label) in enumerate(val_loader):
            image = image.to(torch.float).to(device)
            label = label.squeeze(-1).to(device)
            total_val += label.size(0)

            output = model(image)
            loss = criterion(output, label)
            total_loss_val += loss.item()

            pred = torch.argmax(output, dim=1)
            for i in range(batch_size):
                if pred[i] == label[i]:
                    correct_val += 1
                    
    accuracy_val = 100 * correct_val / total_val
    average_loss_val = total_loss_val / len(val_loader)
    losses['train'].append(average_loss_tr)
    losses['val'].append(average_loss_val)
    
    print("epoch:{} ==Train[loss:{:.5f} acc:{:.2f}%] ==val[loss:{:.5f} acc:{:.2f}%]".format(epoch,
                                                                                            average_loss_tr,accuracy_tr,
                                                                                            average_loss_val,accuracy_val))

    if accuracy_val > acc_best:
        torch.save(model.state_dict(), '../checkpoint/ConvNet.pth')
        print("A new accuracy record has been saved.")
        acc_best = accuracy_val

    df = pd.DataFrame(losses)
    df.to_csv('losses_ConvNet.csv', index=True)

torch.save(model.state_dict(), '../checkpoint/ConvNet_f.pth')