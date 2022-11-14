import sys
sys.path.append('..')
import os
import copy
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from utils.data.custom_finetune_dataset import CustomFinetuneDataset
from utils.data.custom_batch_sampler import CustomBatchSampler
from utils.util import check_dir
from utils.util import save_model

#加载data_root_dir中的数据，返回训练集和验证集的data_loader, data_size
def load_data(data_root_dir):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227,227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])  #transform改变图像会影响标注吗？ 在自定义数据集中，先获取了标注的图像块，然后应用transform
    data_loaders = {}
    for name in ['train', 'val']:
        data_dir = os.path.join(data_root_dir, name)
        dataset = CustomFinetuneDataset(data_dir, transform)
        datasampler = CustomBatchSampler(dataset.get_positive_num(),dataset.get_negative_num(), 32, 96)
        dataloader = DataLoader(dataset, batch_size=128,sampler=datasampler, num_workers=8, drop_last=True)

        data_loaders[name] = dataloader
    return data_loaders

#训练模型
def train_model(data_loaders, model, criterion, optimizer,lr_scheduler, num_epochs=50, device=None):
    best_model_weights = copy.deepcopy(model.state_dict())  #每5代会保存最优都模型
    best_acc = 0        #哪一代效果最好

    epoches = tqdm.trange(num_epochs)
    for epoch in epoches:
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                #梯度置0
                optimizer.zero_grad()

                #forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)


            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / len(data_loaders[phase])
            epoch_acc = running_corrects / len(data_loaders[phase])
            epoches.set_postfix({'name':phase, 'Loss':epoch_loss, 'Acc':epoch_acc}, refresh=False)

            if phase == 'val' and epoch_acc > best_acc and epoch%5==0:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())


    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_weights)
    return model


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
    data_loaders = load_data('../data/finetune_car')

    #迁移学习
    os.environ['TORCH_HOME'] = '../'
    model = models.alexnet(pretrained=True)
    num_features = model.classifier[6].in_features
    #这里可以之间使用 model = AlexNet(num_classes=2)
    model.classifier[6] = nn.Linear(num_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model = train_model(data_loaders, model, criterion,optimizer, lr_scheduler, 50, device)
    check_dir('../models')
    save_model(best_model.state_dict(), '../models/alexnet_car.pth')
    save_model(best_model, '../models/alexnet_car.st')