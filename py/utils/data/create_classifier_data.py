import numpy as np
import shutil
import cv2 as cv
import os
import tqdm
import py.ss
import py.utils.util as utl

#获取正负样本（正样例为所预测框）
#正样本：标注边界框
#负样本：IoU大于0，小于等于0.3。为了进一步限制负样本数目，其大小必须大于标注框的1 / 5
def parse_annotation_jpeg(annotation_path, jpeg_path, gs):
    image = cv.imread(jpeg_path)
    rects = utl.parse_xml(annotation_path)

    py.ss.config(gs, image, 'q')
    bnds = py.ss.get_rects(gs)

    max_size = 0
    for bnd in bnds:
        xmin, ymin, xmax, ymax = bnd
        size = (xmax - xmin) * (ymax - ymin)
        if size > max_size:
            max_size = size

    negative_list = []
    for bnd in bnds:
        xmin, ymin, xmax, ymax = bnd
        size = (xmax - xmin) * (ymax - ymin)
        iou = utl.compute_ious(bnd, rects)
        if iou > 0 and iou <= 0.3 and size > max_size/5:
            negative_list.append(bnd)

    return bnds, negative_list

if __name__ == '__main__':
    car_root_dir = '../../../data/voc_car'
    classifier_root_dir = '../../../data/classifier_car'
    utl.check_dir(classifier_root_dir)

    gs = py.ss.get_selective_search()

    for name in ['train', 'val']:
        src_root_dir = os.path.join(car_root_dir, name)
        src_annotation_dir = os.path.join(src_root_dir, 'Annotations')
        src_jpeg_dir = os.path.join(src_root_dir, 'JPEGImages')

        dst_root_dir = os.path.join(classifier_root_dir, name)
        dst_annotation_dir = os.path.join(dst_root_dir, 'Annotations')
        dst_jpeg_dir = os.path.join(dst_root_dir, 'JPEGImages')

        utl.check_dir(dst_root_dir)
        utl.check_dir(dst_annotation_dir)
        utl.check_dir(dst_jpeg_dir)

        src_car_csv = os.path.join(src_root_dir, 'car.csv')
        dst_car_csv = os.path.join(dst_root_dir, 'car_csv')
        shutil.copy(src_car_csv, dst_car_csv)

        samples = utl.parse_car_csv(src_root_dir)

        total_positive_num = 0
        total_negative_num = 0
        for sample in tqdm.tqdm(samples):
            src_annotation_path = os.path.join(src_annotation_dir, sample + '.xml')
            src_jpeg_path = os.path.join(src_jpeg_dir, sample + 'jpg')
            dst_jpeg_path = os.path.join(dst_jpeg_dir, sample + 'jpg')
            #保存图片
            shutil.copy(src_jpeg_path, dst_jpeg_path)

            #保存样例
            positive_list, negative_list = parse_annotation_jpeg(src_annotation_path, src_jpeg_path, gs)
            total_positive_num += len(positive_list)
            total_negative_num += len(negative_list)
            dst_annotation_positive_path = os.path.join(dst_annotation_dir, sample + '_1.csv')
            dst_annotation_negative_path = os.path.join(dst_annotation_dir, sample + '_0.csv')

            np.savetxt(dst_annotation_positive_path, np.array(positive_list), fmt='%s', delimiter=' ')
            np.savetxt(dst_annotation_negative_path, np.array(negative_list), fmt='%s', delimiter=' ')

        print('positive num :{} in {}'.format(total_negative_num, name))
        print('negative num :{} in {}'.format(total_negative_num, name))


