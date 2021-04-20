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
from network import U_Net

warnings.filterwarnings('ignore', category=FutureWarning)


class Predict():
    def __init__(self, config):
        self.model = U_Net()

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:%i" % config.device[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=config.device)
        self.model = self.model.to(self.device)

        checkpoint = torch.load(config.load_path, map_location=self.device)
        # self.model.module.load_state_dict(checkpoint['state_dict'])
        self.model.load_state_dict(checkpoint['state_dict'])

    def predict(self, loader):
        self.model.eval()
        tbar = tqdm(loader)
        res, features = [], []

        with torch.no_grad():
            for i, (images, _) in enumerate(tbar):
                encode, decoded = self.model(images)
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
    parser.add_argument('--model_name', type=str, default='unet',
                        help='unet_resnet34/unet_se_resnext50_32x4d/unet_efficientnet_b4'
                             '/unet_resnet50/unet_efficientnet_b4')
    # model hyper-parameters
    parser.add_argument('--num_workers', type=int, default=0)
    # dataset
    parser.add_argument('--load_path', type=str, default='./2021-04-19T11-41-25-classify-fold2_best_0.0000480.pth')
    parser.add_argument('--img_path', type=str, default=r'F:\VALLEN\Ni-tension test-electrolysis-1-0.01-AE-20201031\train dataset_wsst')
    config = parser.parse_args()
    print(config)

    loader = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    # loader = transforms.Compose([transforms.ToTensor()])
    unloader = transforms.ToPILImage()
    dataset = []
    for p in os.listdir(config.img_path):
        with open(os.path.join(config.img_path, p), 'rb') as f:
            img_tmp = Image.open(f).convert('RGB')
        dataset.append([loader(img_tmp).unsqueeze(0), 0])

    tmp = Predict(config)
    res, features = tmp.predict(dataset)
    print(np.array(features).shape)
    df = pd.DataFrame(np.array(features))
    df.to_csv('./features_112_112.csv', index=None)

    # for i in tqdm(range(len(dataset))):
    #     fig = plt.figure(figsize=[5.12*2, 5.12], num='1')
    #     ax1 = plt.subplot(121)
    #     ax1.imshow(unloader(dataset[i][0].squeeze(0)))
    #     ax2 = plt.subplot(122)
    #     ax2.imshow(unloader(res[i].squeeze(0).cpu()))
    #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #     plt.margins(0, 0)
    #     plt.show()
    #     # plt.savefig(os.path.join('./res', '%i.jpg' % (i+1)), pad_inches=0)