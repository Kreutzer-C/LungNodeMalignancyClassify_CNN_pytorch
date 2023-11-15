import numpy as np
from skimage import exposure

ct_images=np.load("../dataset/ct_images.npy")

img_rescale=np.zeros(shape=(6691,64,64))

for i in range(ct_images.shape[0]):
    p2, p98 = np.percentile(ct_images[i], (2, 98))
    img_rescale[i] = exposure.rescale_intensity(ct_images[i], in_range=(p2, p98))
    break

np.save("../dataset/ct_images_rescale.npy",img_rescale)
print(img_rescale[0])