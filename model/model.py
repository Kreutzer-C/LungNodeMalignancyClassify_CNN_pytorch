import torch
import torch.nn as nn
from torchvision import models


class ConvNet(nn.Module):
    def __init__(self, num_classes, dropout_prob):
        super(ConvNet, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
        self.relu1 = nn.ReLU()

        # 第二个卷积层
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding='same')
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)

        # 添加 dropout 操作
        self.dropout1 = nn.Dropout(dropout_prob)

        # 第三个卷积层
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.relu3 = nn.ReLU()

        # 第四个卷积层
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)

        # 添加 dropout 操作
        self.dropout2 = nn.Dropout(dropout_prob)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层
        self.fc = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(self.relu2(self.conv2(x)))
        x = self.dropout1(x)  # 添加 dropout
        x = self.relu3(self.conv3(x))
        x = self.maxpool2(self.relu4(self.conv4(x)))
        x = self.dropout2(x)  # 添加 dropout
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #x = self.softmax(x)
        return x