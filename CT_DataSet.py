import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MyCTDataSet(Dataset):
    def __init__(self,images_path,labels_path):
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, item):
        image = self.images[item]
        label=self.labels[item]

        return image,label