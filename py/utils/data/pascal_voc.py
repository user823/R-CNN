import cv2 as cv
import numpy as np
from torchvision.datasets import VOCDetection

if __name__ == '__main__':
    """下载数据集"""
    dataset = VOCDetection('../../../data', year='2007', image_set='trainval', download=True)
