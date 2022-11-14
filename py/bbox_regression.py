import sys
sys.path.append('..')
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils.data.custom_bbox_regression_dataset import BBoxRegressionDataset
import utils.util as util

#只有训练集，没有验证集
def load_data(data_root_dir):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_set = BBoxRegressionDataset(data_root_dir, transform=transform)
    data_loader = DataLoader(data_set, batch_size=128, shuffle=True, num_workers=8)
    return data_loader

def train_model(data_loader, feature_model, model, criterion, optimizer, lr_scheduler, num_epoches=50, device=None):

    model.train()
    loss_list = []
    epoches = tqdm.trange(num_epoches)

    feature_model = feature_model.to(device)
    model = model.to(device)

    for epoch in epoches:

        running_loss = 0.0
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.float().to(device)

            #features_model的features方法其实是网络结构中特征提取层
            features = feature_model.features(inputs)
            features = torch.flatten(features, 1)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            lr_scheduler.step()

        epoch_loss = running_loss / len(data_loader.dataset)
        loss_list.append(epoch_loss)
        epoches.set_postfix({'epoch':epoch, 'loss':epoch_loss})

        #每训练一代就保存下来
        util.save_model(model, '../models/bbox_regression_%d.pth' % epoch)

    return model, loss_list

def get_model():
    model = torch.load('../models/best_linear_svm_alexnet_car.st')
    model.load_state_dict(torch.load('../models/best_linear_svm_alexnet_car.pth'))

    #取消梯度追踪（网络前面特征提取部分不参与训练）
    for param in model.parameters():
        param.requires_grad = False

    return model

if __name__ == '__main__':
    data_loader = load_data('../data/bbox_regression')

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    feature_model = get_model()

    #单独训练网络的位置预测部分
    in_features = 256 * 6 * 6
    out_features = 4
    model = nn.Linear(in_features, out_features)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    model, loss_list = train_model(data_loader, feature_model, model, criterion, optimizer, lr_scheduler, device=device)
    util.save_model(model, '../models/bbox_regression.st')
    util.save_model(model.state_dict(), '../models/bbox_regression.pth')
    util.plot_loss(loss_list)
