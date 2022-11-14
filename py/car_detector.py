import sys
sys.path.append('..')
import time
import copy
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
import ss

import py.utils.util as util

def get_device():
    return torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def get_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform

def get_model(device=None):
    model = torch.load('../models/best_linear_svm_alexnet_car.st')
    model.load_state_dict(torch.load('../models/best_linear_svm_alexnet_car.pth'))
    model.eval()

    #冻结参数
    for param in model.parameters():
        param.requires_grad = False
    if device is not None:
        model = model.to(device)

    return model

def draw_box_with_text(img, rect_list, score_list):
    for i in range(len(rect_list)):
        xmin, ymin, xmax, ymax = rect_list[i]
        score = score_list[i]

        cv.rectangle(img, (xmin,ymin), (xmax,ymax), color=(0,0,255), thickness=1)
        cv.putText(img, '{: .3f}'.format((score)), (xmin,ymin), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

#非极大值抑制
#
#rect_list [N,4],list
#score_list [N],list
def nms(rect_list, score_list):
    nms_rects = []
    nms_scores = []

    rect_array = np.array(rect_list)
    score_array = np.array(score_list)

    #获取从大到小排序后的索引
    idxs = np.argsort(score_array)[::-1]
    rect_array = rect_array[idxs]
    score_array = score_array[idxs]

    thresh = 0.3
    while len(score_array) > 0:
        nms_rects.append(rect_array[0])
        nms_scores.append(score_array[0])
        rect_array = rect_array[1:]
        score_array = score_array[1:]

        if len(score_array) <= 0:
            break

        #计算iou
        iou_scores = util.iou(np.array(nms_rects[len(nms_rects)-1]), rect_array)
        #np.where返回一个元组
        idxs = np.where(iou_scores < thresh)[0]
        rect_array = rect_array[idxs]
        score_array = score_array[idxs]

    return nms_rects, nms_scores


if __name__ == '__main__':
    device = get_device()
    transform = get_transform()
    model = get_model(device)

    #创建selecsearch对象
    gs = ss.get_selective_search()
    test_img_path = '../imgs/000012.jpg'
    test_xml_path = '../imgs/000012.xml'

    img = cv.imread(test_img_path)
    dst = copy.deepcopy(img)

    bndboxs = util.parse_xml(test_xml_path)
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        cv.rectangle(dst, (xmin, ymin), (xmax, ymax), color=(0,255,0), thickness=1)

    #后选区建议
    ss.config(gs, img, strategy='f')
    rects = ss.get_rects(gs)
    print('候选区域建议数目: %d' % len(rects))

    svm_thresh = 0.6
    score_list = []
    positive_list = []

    start = time.time()
    for rect in rects:
        xmin, ymin, xmax, ymax = rect
        rect_img = img[ymin:ymax, xmin:xmax]

        rect_transform = transform(rect_img).to(device)
        output = model(rect_transform.unsqueeze(0))[0]

        if torch.argmax(output).item() == 1:
            #预测为汽车
            probs = torch.softmax(output, dim=0).cpu().numpy()
            if probs[1] >= svm_thresh:
                score_list.append(probs[1])
                positive_list.append(rect)
                print(rect, output, probs)
    end = time.time()
    print('detect time: %d s' % (end-start))
    nms_rects, nms_scores = nms(positive_list, score_list)
    draw_box_with_text(dst, nms_rects, nms_scores)
    cv.imwrite('../imgs/000012_pred.jpg', dst)
