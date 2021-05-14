import argparse
import codecs
import datetime
import json
import os
import pickle
import shutil
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm
from PIL import Image
from network import *

warnings.filterwarnings('ignore', category=FutureWarning)
torch.manual_seed(42)


class UNetMulti3_pre(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(UNetMulti3_pre, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])

        self.feature1 = nn.Conv2d(filters[3], 1, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(14 * 14, 2)
        # self.softmax = nn.Softmax(dim=1)
        self.feature2 = nn.Conv2d(1, filters[3], kernel_size=1, stride=1, padding=0)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.feature1(e4)
        a6 = torch.cat([torch.mean(e5.squeeze(1)[:math.ceil(e5.squeeze(1).size()[0] * 0.6)], dim=0, keepdim=True),
                        torch.mean(e5.squeeze(1)[math.ceil(e5.squeeze(1).size()[0] * 0.6):], dim=0, keepdim=True)], 0)
        a7 = self.linear(a6.flatten(start_dim=1, end_dim=2))

        d5 = self.feature2(e5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return e5.flatten(), a7, out


class Predict():
    def __init__(self, config):
        self.model = UNetMulti3_pre()

        self.device = torch.device("cpu")
        # if torch.cuda.is_available():
        #     self.device = torch.device("cuda:%i" % config.device[0])
        #     self.model = torch.nn.DataParallel(self.model, device_ids=config.device)
        # self.model = self.model.to(self.device)

        checkpoint = torch.load(config.load_path, map_location=self.device)
        # self.model.module.load_state_dict(checkpoint['state_dict'])
        self.model.load_state_dict(checkpoint['state_dict'])

    def predict(self, loader):
        self.model.eval()
        tbar = tqdm(loader)
        res, features = [], []

        with torch.no_grad():
            for i, (images, _) in enumerate(tbar):
                encode, _, decoded = self.model(images)
                # print(encode.shape)
                res.append(decoded)
                features.append(encode.cpu().numpy())

        return res, features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='epoch')
    parser.add_argument("--device", type=int, nargs='+', default=[i for i in range(torch.cuda.device_count())])
    # model set
    parser.add_argument('--model_name', type=str, default='unet_Multi',
                        help='unet_resnet34/unet_se_resnext50_32x4d/unet_efficientnet_b4'
                             '/unet_resnet50/unet_efficientnet_b4')
    # model hyper-parameters
    parser.add_argument('--num_workers', type=int, default=0)
    # dataset
    parser.add_argument('--load_path', type=str, default='./2021-05-13T11-43-18-classify-fold0_best_0.0002364.pth')
    parser.add_argument('--img_path', type=str, default=r'F:\VALLEN\Ni-tension test-pure-1-0.01-AE-20201030\train dataset_wsst')
    config = parser.parse_args()
    print(config)

    loader = transforms.Compose([transforms.Resize(112), transforms.ToTensor()])
    # loader = transforms.Compose([transforms.ToTensor()])
    unloader = transforms.ToPILImage()
    dataset = []
    for p in sorted(os.listdir(config.img_path), key=lambda k: int(k.split('.')[0])):
        with open(os.path.join(config.img_path, p), 'rb') as f:
            img_tmp = Image.open(f).convert('RGB')
        dataset.append([loader(img_tmp).unsqueeze(0), 0])

    tmp = Predict(config)
    res, features = tmp.predict(dataset)
    print(np.array(features).shape)
    df = pd.DataFrame(np.array(features))
    df.to_csv('./Ni_features_112_112_4_GAP_Linear.csv', index=None)

    for i in tqdm(range(len(dataset))):
        fig = plt.figure(figsize=[2.56*2, 2.56], num='1')
        ax1 = plt.subplot(121)
        ax1.imshow(unloader(dataset[i][0].squeeze(0)))
        ax2 = plt.subplot(122)
        ax2.imshow(unloader(res[i].squeeze(0).cpu()))
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.show()
        plt.savefig(os.path.join('./Ni_res_112_112_4_GAP_Linear', '%i.jpg' % (i+1)), pad_inches=0)