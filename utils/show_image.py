import numpy as np
import matplotlib.pyplot as plt
# 4999
# 加载.npy文件
data = np.load('../dataset/ct_images.npy')

image = data[4999]

# 使用matplotlib显示图像
plt.imshow(image, cmap='gray')
plt.show()