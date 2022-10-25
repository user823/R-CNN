import numpy  as np
import os
import cv2 as cv
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import py.utils.util as utl

#定义分类器数据集
#数据集总大小是正例数加负例数(index先索引正例)
class CustomClassifierDataset(Dataset):
    #root_dir为train或者val
    def __init__(self, root_dir, transform=None):
        super().__init__()
        #读取所有汽车样本
        samples = utl.parse_car_csv(root_dir)
        annotation_root_dir = os.path.join(root_dir, 'Annotations')
        img_root_dir = os.path.join(root_dir, 'JPEGImages')

        images = []
        positive_list = []
        negative_list = []

        for index in range(len(samples)):
            sample = samples[index]
            #读取图片
            images.append(cv.imread(img_root_dir, sample + '.jpg'))

            #读取正例
            rects = np.loadtxt(os.path.join(annotation_root_dir, sample + '_1.csv'), dtype=np.int, delimiter=' ')
            for rect in rects:
                positivedic = {}
                positivedic['rect'] = rect
                positivedic['image_id'] = index
                positive_list.append(positivedic)

            #读取负例
            rects = np.loadtxt(os.path.join(annotation_root_dir, sample + '_0.csv'), dtype=np.int, delimiter=' ')
            for rect in rects:
                negativedic = {}
                negativedic['rect'] = rect
                negativedic['image_id'] = index
                negative_list.append(negativedic)

            self.positive_list = positive_list
            self.negative_list = negative_list
            self.transform = transform
            self.images = images


    def __len__(self):
        return len(self.positive_list) + len(self.negative_list)

    def __getitem__(self, index):
        if index < len(self.positive_list):
            target = 1
            dic = self.positive_list[index]

        else:
            target = 0
            index = index - len(self.positive_list)
            dic = self.negative_list[index]

        image_id = dic['image_id']
        rect = dic['rect']     #xmin, ymin, xmax, ymax
        image = self.images[image_id][rect[1]:rect[3], rect[0]:rect[2]]
        if self.transform:
            image = self.transform(image)

        return image, target, dic

    def get_transform(self):
        return self.transform

    def get_positive_num(self):
        return len(self.positive_list)

    def get_negative_num(self):
        return len(self.negative_list)

    def get_positives(self):
        return self.positive_list

    def get_negatives(self):
        return self.negative_list

    def get_jpeg_images(self):
        return self.images