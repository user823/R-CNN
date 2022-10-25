import os
import numpy as np
import xmltodict
import torch
import matplotlib.pyplot as plt

def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

#获取汽车样例,读取到np数组中
def parse_car_csv(csv_dir):
    csv_path = os.path.join(csv_dir, 'car.csv')
    return np.loadtxt(csv_path, dtype=np.str)

#解析xml文件，返回汽车边界框坐标
def parse_xml(xml_path):
    xml=""
    with open(xml_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            xml+=line
    dic = xmltodict.parse(xml)
    bboxes = []
    for object in dic[0]['object']:
        if object['name'] == 'car' and object['difficult'] != 1:
            bbox = object['bndbox']
            bboxes.append((int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])))
    return bboxes

#给定两个边界框计算IOU
#pred_box大小为[4], target_box大小为[N,4]
#返回[1,N]数组
def iou(pred_box, target_box):
    if len(target_box == 1):
        target_box = target_box.reshape((-1, 4))

    xmin = np.maximum(pred_box[0], target_box[:, 0])
    ymin = np.maximum(pred_box[1], target_box[:, 1])
    xmax = np.minimum(pred_box[2], target_box[:, 2])
    ymax = np.minimum(pred_box[3], target_box[:, 3])
    iw = xmax - xmin
    ih = ymax - ymin
    w1 = pred_box[2] - pred_box[0]
    h1 = pred_box[3] - pred_box[1]
    w2 = target_box[:, 2] - target_box[:, 0]
    h2 = target_box[:, 3] - target_box[:, 1]

    intersection = np.maximum(0, iw) * np.maximum(0, ih)
    box1 = w1 * h1
    box2 = w2 * h2
    return intersection / (box1 + box2 - intersection)

#rect是预测框，每个框需要与所有bndbox计算IOU，得分最高的作为rect的得分
def compute_ious(rects, bndboxes):
    score_list = []
    for rect in rects:
        scores = iou(rect, bndboxes)
        score_list.append(max(scores))
    return score_list

#把训练好的模型放在./models/model_path下面
def save_model(model, model_save_path):
    check_dir('./models')
    torch.save(model.state_dict(), model_save_path)

#绘制loss曲线
def plot_loss(loss_list):
    x = np.arange(1,len(loss_list)+1)

    fig, ax = plt.subplots()
    ax.plot(x,loss_list)
    ax.set_title('loss curve')
    plt.savefig('loss curve.png')


if __name__ == '__main__':
    pass