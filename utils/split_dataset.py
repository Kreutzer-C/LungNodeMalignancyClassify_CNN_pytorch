import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ct_images=np.load("../dataset/ct_images_rescale.npy")
labels=pd.read_csv("../dataset/mal2.csv")

# 划分数据集为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(ct_images, labels, test_size=0.2, random_state=42)

X_train=X_train.reshape(5352,1,64,64)
X_val=X_val.reshape(1339,1,64,64)

# 保存训练集和验证集到新的 Numpy 文件
np.save('../dataset/train_images.npy', X_train)
np.save('../dataset/val_images.npy', X_val)
np.save('../dataset/train_labels.npy', y_train)
np.save('../dataset/val_labels.npy', y_val)