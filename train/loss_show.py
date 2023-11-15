import pandas as pd
import matplotlib.pyplot as plt

# 读取包含train_loss和val_loss的CSV文件
df = pd.read_csv('losses_ConvNet_11.csv')

# 获取训练集和验证集的损失数据
train_loss = df['train']
val_loss = df['val']

# 创建x轴数据，即epoch数
epochs = range(1, len(train_loss) + 1)

# 绘制训练集和验证集损失曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')

# 添加标题和标签
plt.title('Train and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# 添加图例
plt.legend()

# 显示网格
plt.grid()

# 显示损失曲线图
plt.show()
