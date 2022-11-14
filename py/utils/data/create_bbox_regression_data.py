import py.utils.util as utl
import numpy as np
import os
import shutil
import tqdm

#创建边界框回归数据集

#从voc_car/train目录中提取标注边界框坐标
#从finetune_car/train目录中提取训练集正样本坐标（IoU>=0.5），进一步提取IoU>0.6的边界框
#数据集保存在bbox_car目录下

if __name__ == '__main__':
    voc_car_train_dir = '../../../data/voc_car/train'
    gt_annotation_dir = os.path.join(voc_car_train_dir, 'Annotations')  #xml文件封装
    jpeg_dir = os.path.join(voc_car_train_dir, 'JPEGImages')

    classifier_car_train_dir = '../../../data/finetune_car/train'
    positive_annotation_dir = os.path.join(classifier_car_train_dir, 'Annotations')

    dst_root_dir = '../../../data/bbox_regression'
    dst_jpeg_dir = os.path.join(dst_root_dir, 'JPEGImages')
    dst_bndbox_dir = os.path.join(dst_root_dir, 'bndboxs')    #所有gt
    dst_positive_dir = os.path.join(dst_root_dir, 'positive') #所有iou>0.6

    utl.check_dir(dst_root_dir)
    utl.check_dir(dst_jpeg_dir)
    utl.check_dir(dst_bndbox_dir)
    utl.check_dir(dst_positive_dir)

    samples = utl.parse_car_csv(voc_car_train_dir)
    saved_sample = []
    total_positive_num = 0
    for sample in tqdm.tqdm(samples):
        annotation_path = os.path.join(gt_annotation_dir, sample + '.xml')
        bnds = utl.parse_xml(annotation_path)

        positive_annotation_path = os.path.join(positive_annotation_dir, sample + '_1.csv')
        positive_bnds = np.loadtxt(positive_annotation_path, dtype=np.int64, delimiter=' ')

        positive_list = []
        #搜索iou>0.6的微调数据集预测框
        #处理单行单情况
        if len(positive_bnds.shape) == 1:
            positive_bnds = np.array([positive_bnds])
        for positive_bnd in positive_bnds:
            ious = utl.iou(positive_bnd, bnds)
            if max(ious) > 0.6:
                positive_list.append(positive_bnd)

        if len(positive_list) > 0:
            # 搬运图片
            src_jpeg_path = os.path.join(jpeg_dir, sample + '.jpg')
            dst_jpeg_path = os.path.join(dst_jpeg_dir, sample + '.jpg')
            shutil.copy(src_jpeg_path, dst_jpeg_path)

            dst_bndbox_path = os.path.join(dst_bndbox_dir, sample + '.csv')
            np.savetxt(dst_bndbox_path, bnds, fmt='%s', delimiter=' ')

            dst_positive_path = os.path.join(dst_positive_dir, sample + '.csv')
            np.savetxt(dst_positive_path, np.array(positive_list), fmt='%s', delimiter=' ')

            total_positive_num += len(positive_list)
            saved_sample.append(sample)

    print('{} images has been saved'.format(total_positive_num))
    dst_car_csv_path = os.path.join(dst_root_dir, 'car.csv')
    np.savetxt(dst_car_csv_path, np.array(saved_sample), fmt='%s', delimiter=' ')

