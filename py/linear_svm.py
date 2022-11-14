import sys
sys.path.append('..')
import tqdm
import copy
import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils.data.custom_classifier_dataset import CustomClassifierDataset
from utils.data.custom_hard_negative_mining_dataset import CustomHardNegativeMiningDataset
from utils.data.custom_batch_sampler import CustomBatchSampler
from utils.util import check_dir
from utils.util import save_model

batch_positive = 32
batch_negative = 96
batch_total = 128

#创建训练集和验证集的data_loader
#data_loader['remain']存放剩下的negative_list(image_id,rect)，初始化之后在后续步骤不再改变
def load_data(data_root_dir):
    transform = transforms.Compose([
         transforms.ToPILImage(),
         transforms.Resize((227,227)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_loaders = {}
    for name in ['train','val']:
        root_dir = os.path.join(data_root_dir, name)
        data_set = CustomClassifierDataset(root_dir, transform)

        #使用负难例挖掘的方式训练，初始化正负样本
        if name == 'train':
            positive_list = data_set.get_positives()
            negative_list = data_set.get_negatives()

            init_negative_idxs = random.sample(range(len(negative_list)), len(positive_list))
            init_negative_list = [negative_list[idx] for idx in init_negative_idxs]
            remain_nagetive_list = [negative_list[idx] for idx in range(len(negative_list)) if idx not in init_negative_idxs]

            data_set.set_negative_list(init_negative_list)
            data_loaders['remain'] = remain_nagetive_list

        sampler = CustomBatchSampler(data_set.get_positive_num(), data_set.get_negative_num(), batch_positive, batch_negative)
        data_loader = DataLoader(data_set, batch_size=batch_total, sampler=sampler, num_workers=8, drop_last=True)
        data_loaders[name] = data_loader

    return data_loaders

#hinge 折页
#outputs(N * 2)   labels(N)每行取值0/1
#return loss
def hinge_loss(outputs, labels):
    num_labels = len(labels)     # N
    #把outputs对应labels出的值提取处理并且变成二维数组 (N,1)
    corrects = outputs[range(num_labels), labels].unsqueeze(0).T

    #最大间隔
    margin = 1.0
    margins = outputs - corrects + margin
    loss = torch.sum(torch.max(margins, 1)[0]) / num_labels

    return loss

#调整训练集的negative_list
#add_negative_list用来记录添加过的难例，以后不再添加
def add_hard_negatives(hard_negative_list, negative_list, add_negative_list):
    for item in hard_negative_list:
        if len(add_negative_list) == 0:
            #第一次添加负样本
            negative_list.append(item)
            add_negative_list.append(list(item['rect']))     #为什么加list
        if list(item['rect']) not in add_negative_list:
            negative_list.append(item)
            add_negative_list.append(list(item['rect']))

#返回难例（假阳性）rect 和 image_id
#返回简单例（真阴性）rect 和 image_id
#preds: N * 1
#cache_dicts: N * 1
def get_hard_negatives(preds, cache_dicts):
    fp_mask = preds == 1    #假阳性的索引
    tn_mask = preds == 0    #真阴性的索引

    fp_rects = cache_dicts['rect'][fp_mask].numpy()
    fp_image_ids = cache_dicts['image_id'][fp_mask].numpy()

    tn_rects = cache_dicts['rect'][tn_mask].numpy()
    tn_image_ids = cache_dicts['image_id'][tn_mask].numpy()

    hard_negative_list = [{'rect':fp_rects[idx], 'image_id':fp_image_ids[idx]} for idx in range(len(fp_rects))]
    easy_negatie_list = [{'rect': tn_rects[idx], 'image_id': tn_image_ids[idx]} for idx in range(len(tn_rects))]

    return hard_negative_list, easy_negatie_list

def train_model(data_loaders, model, criterion, optimizer, lr_scheduler, num_epoches=50, device=None):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epoches = tqdm.trange(num_epoches)
    for epoch in epoches:
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            #输出正负样本数
            data_set = data_loaders[phase].dataset
            print('{} - positive_num: {} - negative_num: {} - data size: {}'.format(
                phase, data_set.get_positive_num(), data_set.get_negative_num(), len(data_set)))

            for inputs, labels, cache_dicts in data_loaders[phase]:
                inputs = inputs.to(device)          #inputs: batch_size * 227 * 227
                labels = labels.to(device)          #labels：batch_size * 1

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)         #outputs: batch_size * 2
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / len(data_set)
            epoch_acc = running_corrects / len(data_set)

            epoches.set_postfix({'name':phase, 'Loss':epoch_loss, 'Acc':epoch_acc})

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())


        #前面部分都是完成正常的训练， 下面会对剩余样本集进行负难例挖掘
        train_dataset = data_loaders['train'].dataset
        remain_negative_list = data_loaders['remain']
        jpeg_images = train_dataset.get_jpeg_images()
        transform = train_dataset.get_transform()  #保持变换的一致性

        with torch.set_grad_enabled(False):
            #创建remain_negative_list的data_loader
            remain_dataset = CustomHardNegativeMiningDataset(remain_negative_list, jpeg_images, transform=transform)
            remain_data_loader = DataLoader(remain_dataset, batch_size=batch_total, num_workers=8, drop_last=True)

            #获取训练数据集的负样本集
            negative_list = train_dataset.get_negatives()
            #add_negative用来保存添加过的难例rect，保证每个难例只添加一次
            add_negative_list = data_loaders.get('add_negative', []) #字典的get方法可以在键不存在时返回指定值

            running_corrects = 0.0
            #剩余集全部是负例，但是因为误差可以区分为假阳性（难例）和真阴性（简单例）
            for inputs, labels, cache_dicts in remain_data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                running_corrects += torch.sum(preds == labels)
                hard_negative_list, easy_negative_list = get_hard_negatives(preds.cpu().numpy(), cache_dicts)
                add_hard_negatives(hard_negative_list, negative_list, add_negative_list)

            #remain_acc 可以表示假阳率
            remain_acc = running_corrects / len(remain_negative_list)
            print('remain negative size: {}, acc: {:.4f}'.format(len(remain_negative_list), remain_acc))

            #训练完后重置负样本，进行难负例挖掘
            train_dataset.set_negative_list(negative_list)
            tmp_sampler = CustomBatchSampler(train_dataset.get_positive_num(), train_dataset.get_negative_num(),
                                             batch_positive, batch_negative)
            data_loaders['train'] = DataLoader(train_dataset, batch_size=batch_total, sampler=tmp_sampler,
                                               num_workers=8, drop_last=True)
            data_loaders['add_negative'] = add_negative_list

        #每训练完一轮就保存
        save_model(model, '../models/linear_svm_alexnet_car_%d.pth' % epoch)

    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_weights)
    return model

if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    data_loaders = load_data('../data/classifier_car')

    #加载CNN模型
    model_state_path = '../models/alexnet_car.pth'
    model_struct_path = '../models/alexnet_car.st'
    model = torch.load(model_struct_path)
    model.load_state_dict(torch.load(model_state_path))
    model = model.to(device)

    criterion = hinge_loss
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    #共训练10轮，每4轮减少一次学习率
    lr_schduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    best_model = train_model(data_loaders, model, criterion, optimizer, lr_schduler, num_epoches=10, device=device)
    #保存最好的模型参数
    save_model(best_model.state_dict(), '../models/best_linear_svm_alexnet_car.pth')
    save_model(best_model, '../models/best_linear_svm_alexnet_car.st')
