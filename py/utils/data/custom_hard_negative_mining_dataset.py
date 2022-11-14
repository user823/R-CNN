import torch.nn as nn
from torch.utils.data import Dataset
from py.utils.data.custom_classifier_dataset import CustomClassifierDataset

#只索引图片中的负例框,并应用transform
class CustomHardNegativeMiningDataset(Dataset):
    def __init__(self, negative_list, jpeg_images, transform= None):
        super().__init__()
        self.negative_list = negative_list
        self.jpeg_images = jpeg_images
        self.transform = transform

    def __getitem__(self, index):
        target = 0

        negative_dic = self.negative_list[index]
        xmin, ymin, xmax, ymax = negative_dic['rect']
        image_id = negative_dic['image_id']

        image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]
        if self.transform:
            image = self.transform(image)

        return image, target, negative_dic

    def __len__(self):
        return len(self.negative_list)

