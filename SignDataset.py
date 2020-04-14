import os

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader


class SignDataset(Dataset):

    #def __init__(self, csv_file, root_dir, transform=None):
    def __init__(self):

        print('initializing dataset')

        self.labels = np.loadtxt('.\\data\\training_labels\\labels.csv')

        self.image_names = os.listdir('.\\data\\training_images\\combined')
        self.image_names.sort()
        self.image_names = [os.path.join('.\\data\\training_images\\combined', img_name) for img_name in self.image_names]

        print(self.image_names)
        print(self.labels)

        #self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        letter = self.labels[idx]
        print(self.image_names[idx])
        #image = Image.open(self.image_names[idx])
        image = io.imread(self.image_names[idx])
        return letter, image


#            if(torch.is_tensor(idx)):
#                idx = idx.tolist()
#
#                img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx,0])
#                image = io.imread(img_name)
#                landmarks = self.landmarks_frame.iloc[idx,0]

# dataset = SignDataset()
# images = []
# for n in dataset.image_names:
#     print("filepath: " + n)
#     images.append(np.array(Image.open(n)))
# a, b = dataset[0]
#
# print("a: {}".format(a))
# b.show()