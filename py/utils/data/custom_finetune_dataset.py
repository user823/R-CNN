import numpy as np
import os
import cv2 as cv
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import py.utils.util as utl

#读取训练集或验证集的正例
class CustomFinetuneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        samples = utl.parse_car_csv(root_dir)

        jpeg_images = []
        #边界框的坐标
        positive_rects = []
        negative_rects = []

        for idx in range(len(samples)):
            sample = samples[idx]
            jpeg_images.append(cv.imread(os.path.join(root_dir, 'JPEGImages', sample + '.jpg')))
            bnds = np.loadtxt(os.path.join(root_dir, 'Annotations', sample + '_1.csv'), dtype=np.int64, delimiter=' ')
            #如果只有一行的话bnds，用for遍历bnds将得到四个值
            if len(bnds.shape) == 1:
                bnds = np.array([bnds])
            for bnd in bnds:
                positivedic = {}
                positivedic['rect'] = bnd
                positivedic['image_id'] = idx
                positive_rects.append(positivedic)

            bnds = np.loadtxt(os.path.join(root_dir, 'Annotations', sample + '_0.csv'), dtype=np.int64, delimiter=' ')
            if len(bnds.shape) == 1:
                bnds = np.array([bnds])
            for bnd in bnds:
                negativedic = {}
                negativedic['rect'] = bnd
                negativedic['image_id'] = idx
                negative_rects.append(negativedic)


        self.images = jpeg_images
        self.transform = transform
        self.positive_rects = positive_rects
        self.negative_rects = negative_rects

    def __len__(self):
        return len(self.positive_rects) + len(self.negative_rects)

    def __getitem__(self, index):
        if index < len(self.positive_rects):
            dic = self.positive_rects[index]
            target = 1
        else:
            index = index - len(self.positive_rects)
            dic = self.negative_rects[index]
            target = 0
        image = self.images[dic['image_id']]
        xmin, ymin, xmax, ymax = dic['rect']
        image = image[ymin:ymax, xmin:xmax]
        if self.transform:
            image = self.transform(image)

        return image, target

    def get_positive_num(self):
        return len(self.positive_rects)

    def get_negative_num(self):
        return len(self.negative_rects)


if __name__ == '__main__':
    dataset = CustomFinetuneDataset('../../../data/finetune_car/train')
    data_loader = DataLoader(dataset)
    for inputs, target in data_loader:
        pass