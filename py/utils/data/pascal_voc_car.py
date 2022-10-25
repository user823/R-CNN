import os
import shutil
import random
import numpy as np
import py.utils.util as utl

#定义文件后缀，因为路径文件中只保存了文件名字
suffix_xml ='.xml'
suffix_jpeg = '.jpg'

dataset_root_dir = '../../../data'

car_train_path = dataset_root_dir + '/VOCdevkit/VOC2007/ImageSets/Main/car_train.txt'
car_val_path = dataset_root_dir + '/VOCdevkit/VOC2007/ImageSets/Main/car_val.txt'

#设置标注和图片的根目录
voc_annotation_dir = dataset_root_dir + '/VOCdevkit/VOC2007/Annotations/'
voc_jpeg_dir = dataset_root_dir + '/VOCdevkit/VOC2007/JPEGImages/'

#处理汽车图片之后的存放路径
car_root_dir = dataset_root_dir + '/voc_car'

#提取指定的图像类别,返回np数组
def parse_train_val(data_path):
    samples = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            res = line.strip().split(' ')
            if len(res[0])==3 and int(res[1])==1:
                samples.append(res[0])
    return np.array(samples)

#对训练集和验证集进行随机采样，保留1/10对数据
def sample_train_val(samples):
    for name in ['train','val']:
        dataset = samples[name]
        length = len(dataset)
        dataset_samples = random.sample(dataset, length//10)
        samples[name]= dataset_samples

#保存类别car的样本图片和标注文件到data_annotation_dir,data_jpeg_dir中
def save_car(car_samples, data_root_dir, data_annotation_dir, data_jpeg_dir):
    for sample in car_samples:
        src_annotation_path = os.path.join(voc_annotation_dir, sample + suffix_xml)
        dst_annotation_path = os.path.join(data_annotation_dir, sample + suffix_xml)
        shutil.copy(src_annotation_path, dst_annotation_path)

        src_jpg_path = os.path.join(voc_jpeg_dir, sample + suffix_jpeg)
        dst_jpg_path = os.path.join(data_jpeg_dir, sample + suffix_jpeg)
        shutil.copy(src_jpg_path, dst_jpg_path)

    #train 和 val 各自有一个csv文件
    csv_path = os.path.join(data_root_dir, 'car.csv')
    np.savetxt(csv_path, np.array(car_samples), fmt='%s')

if __name__ == '__main__':
    #读取汽车图片的训练集和验证集样本
    samples = {'train':parse_train_val(car_train_path), 'val':parse_train_val(car_val_path)}
    utl.check_dir(car_root_dir)       #建立汽车根目录

    #sample_train_val(samples)   #减少样本数目

    #把训练集和验证集都搬到car_root_dir中
    for name in ['train', 'val']:
        data_root_dir = os.path.join(car_root_dir, name)   #首先区分数据集和验证集
        car_annotation_dir = os.path.join(data_root_dir, 'Annotations')     #标注文件
        car_jpeg_dir = os.path.join(data_root_dir, 'JPEGImages')           #图片文件
        utl.check_dir(data_root_dir)
        utl.check_dir(car_annotation_dir)
        utl.check_dir(car_jpeg_dir)
        save_car(samples[name], data_root_dir, car_annotation_dir, car_jpeg_dir)

    print('car_root has created.')
