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
    data_sizes = {}
    for name in ['train', 'val']:
        data_dir = os.path.join(data_root_dir, name)
        dataset = CustomFinetuneDataset(data_dir, transform)
        datasampler = CustomBatchSampler(dataset.get_positive_num(),dataset.get_negative_num(), 32, 96)
        dataloader = DataLoader(dataset, batch_size=128,sampler=datasampler, num_workers=8, drop_last=True)

        data_loaders[name] = dataloader
        data_sizes[name] = len(datasampler)
    return data_loaders, data_sizes

#训练模型
def train_model(data_loaders, model, criterion, optimizer,lr_scheduler, num_epochs=100, device=None):
    best_model_weights = copy.deepcopy(model.state_dict())  #每5代会保存最优都模型
    best_acc = 0

    for epoch in tqdm.trange(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0


