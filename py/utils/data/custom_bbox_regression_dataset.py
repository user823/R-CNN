import os
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import py.utils.util as util

#组合(image_id, bndbox, positive)
class BBoxRegressionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.transform = transform

        samples = util.parse_car_csv(root_dir)
        jpeg_list = []
        box_list = []

        #i表示图片的id
        for i in range(len(samples)):
            sample = samples[i]

            jpeg_path = os.path.join(root_dir, 'JPEGImages', sample + '.jpg')
            bndbox_path = os.path.join(root_dir, 'bndboxs', sample + '.csv')
            positive_path = os.path.join(root_dir, 'positive', sample + '.csv')

            jpeg_list.append(cv.imread(jpeg_path))
            bndboxs = np.loadtxt(bndbox_path, dtype=np.int64, delimiter=' ')
            positives = np.loadtxt(positive_path, dtype=np.int64, delimiter=' ')

            if len(positives.shape) == 1:
                bndbox = self.get_bndbox(bndboxs, positives)
                box_list.append({'image_id':i, 'bndbox': bndbox, 'positive':positives})

            else:
                for positive in positives:
                    bndbox = self.get_bndbox(bndboxs, positive)
                    box_list.append({'image_id':i, 'bndbox': bndbox, 'positive':positive})

        self.jpeg_list = jpeg_list
        self.box_list = box_list

    #获取预测框图像，目标中心点相对位置和目标宽高的相对长度
    #t_x = (g_x - p_x)/p_w   t_y = (g_y-p_y)/p_h    它们都是被归一化的相对位置
    #t_w = log(g_w) - log(p_w)   t_h = log(g_h) - log(p_h)
    def __getitem__(self, index):
        dic = self.box_list[index]
        image_id = dic['image_id']
        bndbox = dic['bndbox']
        positive = dic['positive']

        jpeg_img = self.jpeg_list[image_id]
        xmin, ymin, xmax, ymax = positive
        image = jpeg_img[ymin:ymax, xmin:xmax]

        if self.transform is not None:
            image = self.transform(image)

        #计算预测中心点
        p_w = xmax - xmin
        p_h = ymax - ymin
        p_x = xmin + p_w/2
        p_y = ymin + p_h/2

        #计算标注框中心点
        xmin, ymin, xmax, ymax = bndbox
        g_w = xmax - xmin
        g_h = ymax - ymin
        g_x = xmin + g_w/2
        g_y = ymin + g_h/2

        t_x = (g_x - p_x) / p_w
        t_y = (g_y - p_y) / p_h
        t_w = np.log(g_w / p_w)
        t_h = np.log(g_h / p_h)

        return image, np.array((t_x, t_y, t_w, t_h))

    def __len__(self):
        return len(self.box_list)

    #对于同一个id的图片，返回和预测框positive的iou最大的标注框
    def get_bndbox(self, bndboxes, positive):
        if len(bndboxes.shape) == 1:
            return bndboxes
        else:
            ious = util.iou(positive, bndboxes)
            return bndboxes[np.argmax(ious)]

if __name__ == '__main__':
    # util.rename('../../../data/bbox_regression/JPEGImages','jpg')
    dataset = BBoxRegressionDataset('../../../data/bbox_regression')
    data_loader = DataLoader(dataset)
    for inputs, targets in data_loader:
        pass