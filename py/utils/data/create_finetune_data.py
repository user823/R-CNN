import shutil
import numpy as np
import cv2 as cv
import os
import tqdm
import py.ss
import py.utils.util as utl

#获取正负样本的边界框
#正样本：标注边界框
#负样本：IoU大于0，小于等于0.5。为了进一步限制负样本数目，其大小必须大于标注框的1/5

def parse_annotation_jpeg(annotation_path, jpeg_path, gs):
    positive_list = []
    negative_list = []

    img = cv.imread(jpeg_path)
    py.ss.config(gs, img)       #使用质量模式生成候选框
    rects = py.ss.get_rects(gs)
    rects = np.array(rects)

    bnds = utl.parse_xml(annotation_path)
    bnds = np.array(bnds)
    # print(bnds)     #[[156  97 351 270]]

    maximum_box_size = 0        #找到最大标注框
    for bnd in bnds:
        xmin, ymin, xmax, ymax = bnd
        rect_size = (xmax - xmin) * (ymax - ymin)
        if rect_size > maximum_box_size:
            maximum_box_size = rect_size

    iou_list = utl.compute_ious(rects, bnds)
    for idx in range(len(rects)):
        rect = rects[idx]
        iou = iou_list[idx]
        xmin, xmax, ymin, ymax = rect
        if iou > 0.5:
            positive_list.append(rect)
        elif iou > 0:
            rect_size = (xmax - xmin) * (ymax - ymin)
            if rect_size > maximum_box_size / 5:
                negative_list.append(rect)

    return positive_list, negative_list

if __name__ == '__main__':
    car_root_dir = '../../../data/voc_car'
    dst_root_dir = '../../../data/finetune_car'
    utl.check_dir(dst_root_dir)

    gs = py.ss.get_selective_search()

    for name in ['train','val']:
        dst_root_dir_name = os.path.join(dst_root_dir, name)
        src_annotation_dir = os.path.join(car_root_dir, name, 'Annotations')
        src_jpeg_dir = os.path.join(car_root_dir, name, 'JPEGImages')
        dst_annotation_dir = os.path.join(dst_root_dir_name, 'Annotations')
        dst_jpeg_dir = os.path.join(dst_root_dir_name, 'JPEGImages')

        utl.check_dir(dst_root_dir_name)
        utl.check_dir(dst_annotation_dir)
        utl.check_dir(dst_jpeg_dir)

        src_car_csv = os.path.join(car_root_dir, name, 'car.csv')         #训练集和验证集都有自己的csv文件
        dst_car_csv = os.path.join(dst_root_dir_name, 'car.csv')
        shutil.copy(src_car_csv, dst_car_csv)                             #直接搬运voc_car的csv文件
        samples = utl.parse_car_csv(dst_root_dir_name)

        total_positive = 0
        total_negative = 0

        for sample in tqdm.tqdm(samples):
            annotation_path = os.path.join(src_annotation_dir, str(sample) + '.xml')
            jpeg_path = os.path.join(src_jpeg_dir, str(sample) + '.jpg')
            positive_list, negative_list = parse_annotation_jpeg(annotation_path, jpeg_path, gs)
            total_positive += len(positive_list)
            total_negative += len(negative_list)

            #保存标注
            positive_annotation_path = os.path.join(dst_annotation_dir, sample + '_1' + '.csv')
            negative_annotation_path = os.path.join(dst_annotation_dir, sample + '_0' + '.csv')
            np.savetxt(positive_annotation_path, np.array(positive_list), fmt='%s', delimiter=' ')
            np.savetxt(negative_annotation_path, np.array(negative_list), fmt='%s', delimiter=' ')

            #搬运图片
            shutil.copy(jpeg_path, os.path.join(dst_jpeg_dir, str(sample) + '.jpg'))

        print('total_positive: {} in {}'.format(total_positive, name))
        print('total_negative: {} in {}'.format(total_negative, name))