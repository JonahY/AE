import os
import numpy as np
import math
import time
import argparse
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import torchvision
from torchvision import models
from solver import Solver
from PIL import Image
import torchvision.transforms as transforms


class Predict():
    def __init__(self, config):
        self.model = models.shufflenet_v2_x1_0(pretrained=False)
        self.model.fc = nn.Sequential(nn.Linear(1024, config.class_num), nn.Sigmoid())

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:%i" % config.device[0])
            # self.model = torch.nn.DataParallel(self.model, device_ids=config.device)
        self.model = self.model.to(self.device)

        self.weight_path = config.weight_path
        self.solver = Solver(self.model, self.device)

    def predict(self, dataset):
        self.model.eval()
        self.model.train(False)
        checkpoint = torch.load(self.weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        other_idx = []

        with torch.no_grad():
            for i, (images, _) in tqdm(enumerate(dataset)):
                labels_predict = self.solver.forward(images)
                labels_predict = (labels_predict > 0.8).float()
                for o_idx, tmp in enumerate(labels_predict):
                    if not any(tmp):
                        other_idx.append(o_idx + i * images.shape[0])
                labels_predict = torch.max(labels_predict, 1)[1].cpu()
                pre_res = labels_predict if not i else np.concatenate((pre_res, labels_predict))

        return pre_res, other_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--device", type=int, nargs='+', default=[i for i in range(torch.cuda.device_count())])
    parser.add_argument('--class_num', type=int, default=3)
    parser.add_argument('--weight_path', type=str,
                        default='/home/Yuanbincheng/data/Ni-tension test-pure-1-0.01-AE-20201030/checkpoints/'
                                'shufflenet/2021-03-22T20-43-29-classify/2021-03-22T20-43-29-classify-fold6/'
                                '2021-03-22T20-43-29-classify-fold6_best_1.0000.pth')
    parser.add_argument('--predict_fold', type=str,
                        default='/home/Yuanbincheng/data/Ni-tension test-pure-1-0.01-AE-20201030/'
                                'train dataset_cwt_-noise')
    config = parser.parse_args()

    # ls = sorted(os.listdir(config.predict_fold), key=lambda k: int(k.split('.')[0]))
    # pre_dataset = np.expand_dims(np.array(Image.open(os.path.join(config.predict_fold, ls[0]))), 0)
    # for i in tqdm(os.listdir(config.predict_fold)[1:]):
    #     tmp = np.expand_dims(np.array(Image.open(os.path.join(config.predict_fold, i))), 0)
    #     pre_dataset = np.concatenate((tmp, pre_dataset))
    # pre_dataset = torch.from_numpy(pre_dataset).permute(0, 3, 1, 2)

    pre_dataset = torchvision.datasets.ImageFolder(root=config.predict_fold,
                                                   transform=transforms.Compose([transforms.Resize(224),
                                                                                 transforms.ToTensor()]))
    pre_loader = DataLoader(pre_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    train_val = Predict(config)
    pre_res, other_idx = train_val.predict(pre_loader)

    # pre_res[other_idx] = [-1] * len(other_idx)
    # idx_1 = np.where(pre_res == 0)[0]
    # idx_2 = np.where(pre_res == 1)[0]
    # idx_3 = np.where(pre_res == 2)[0]
    # idx_other = np.where(pre_res == -1)[0]
