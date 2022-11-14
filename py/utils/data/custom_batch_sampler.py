import numpy as np
import random
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from py.utils.data.custom_finetune_dataset import CustomFinetuneDataset


#自定义采样器(一般需要配合dataset的定义）
class CustomBatchSampler(Sampler):
    #每次批处理，num_positive:正样本数 num_negative:负样本数目
    #batch_positive:单次正样本数目, batch_negative:单次负样本数目
    def __init__(self,num_positive, num_negative, batch_positive, batch_negative):
        super().__init__(None)
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.batch_positive = batch_positive
        self.batch_negative = batch_negative

        length = num_positive + num_negative
        self.idx_list = list(range(length))

        self.batch = batch_negative + batch_positive
        self.num_batch = length // self.batch

    def __iter__(self):
        sampler_list = []
        for i in range(self.num_batch):
            tmp = np.concatenate(
                (random.sample(self.idx_list[:self.num_positive], self.batch_positive),
                random.sample(self.idx_list[self.num_positive:], self.batch_negative))
            )
            random.shuffle(tmp)
            sampler_list.extend(tmp)
        return iter(sampler_list)

    def __len__(self):
        return self.num_batch * self.batch

    def get_num_batch(self):
        return self.num_batch