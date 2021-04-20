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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm
from network import U_Net

warnings.filterwarnings('ignore', category=FutureWarning)


class TrainVal():
    def __init__(self, config):
        self.model = U_Net()

        # # freeze model parameters
        # for param in self.model.parameters():
        #     param.requires_grad = False
        #
        # # model check
        # print(self.model)
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print("requires_grad: True ", name)
        #     else:
        #         print("requires_grad: False ", name)

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:%i" % config.device[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=config.device)
        self.model = self.model.to(self.device)

        if config.load_path:
            checkpoint = torch.load(config.load_path, map_location=self.device)
            self.model.module.load_state_dict(checkpoint['state_dict'])

        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.epoch = config.epoch
        self.splits = config.n_splits
        self.root = config.root

        self.criterion = nn.MSELoss()

        self.TIME = "{0:%Y-%m-%dT%H-%M-%S}-classify".format(datetime.datetime.now())
        self.model_path = os.path.join(config.root, config.save_path, config.model_name, self.TIME)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.min_loss = float('inf')
        self.seed = int(time.time())
        # self.seed = 1570421136
        # seed_torch(self.seed)

        self.train_transform = transforms.Compose(
            [transforms.Resize([256, 256]),
             transforms.RandomCrop(224),
             transforms.RandomRotation(degrees=(-40, 40)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()])
        self.test_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    def train(self, create_data=False):
        if create_data:
            df = pd.read_csv(os.path.join(self.root, 'train info_wsst.csv'), header=None)
            labels_1dim = np.argmax(np.array(df), axis=1)
            print('<' * 20 + ' Start creating datasets ' + '>' * 20)
            skf = StratifiedKFold(n_splits=self.splits, shuffle=True, random_state=55)
            for idx, [train_df_index, val_df_index] in tqdm(enumerate(skf.split(df, labels_1dim), 1)):
                for i in train_df_index:
                    try:
                        shutil.copy(os.path.join(config.root, 'train dataset_wsst/%s.jpg' % (i + 1)),
                                    os.path.join('/home/Yuanbincheng/project/dislocation_cls/3/SAE',
                                                 'train_%d/%d/%d.jpg' % (idx, labels_1dim[i], i + 1)))
                    except FileNotFoundError:
                        try:
                            os.mkdir(
                                os.path.join('/home/Yuanbincheng/project/dislocation_cls/3/SAE', 'train_%d' % idx))
                        except (FileNotFoundError, FileExistsError):
                            os.mkdir(os.path.join('/home/Yuanbincheng/project/dislocation_cls/3/SAE',
                                                  'train_%d/%d' % (idx, labels_1dim[i])))
                for i in val_df_index:
                    try:
                        shutil.copy(os.path.join(config.root, 'train dataset_wsst/%s.jpg' % (i + 1)),
                                    os.path.join('/home/Yuanbincheng/project/dislocation_cls/3/SAE',
                                                 'test_%d/%d/%d.jpg' % (idx, labels_1dim[i], i + 1)))
                    except FileNotFoundError:
                        try:
                            os.mkdir(os.path.join('/home/Yuanbincheng/project/dislocation_cls/3/SAE', 'test_%d' % idx))
                        except (FileNotFoundError, FileExistsError):
                            os.mkdir(os.path.join('/home/Yuanbincheng/project/dislocation_cls/3/SAE',
                                                  'test_%d/%d' % (idx, labels_1dim[i])))
            print('<' * 20 + ' Finish creating datasets ' + '>' * 20)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.module.parameters()), self.lr,
                               weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.99)
        global_step, global_threshold, global_threshold_pop1, global_threshold_pop2, global_threshold_pop3 = 1, 1, 1, 1, 1

        for fold_index in range(1):
            train_dataset = torchvision.datasets.ImageFolder(root='SAE/train_%d/' % (fold_index + 1),
                                                             transform=self.train_transform)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                      num_workers=config.num_workers)
            val_dataset = torchvision.datasets.ImageFolder(root='SAE/train_%d/' % (fold_index + 1),
                                                           transform=self.test_transform)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                    num_workers=config.num_workers)
            self.model.train()

            TIMESTAMP = '-fold'.join([self.TIME, str(fold_index)])
            self.writer = SummaryWriter(log_dir=os.path.join(self.model_path, TIMESTAMP))
            with codecs.open(os.path.join(self.model_path, TIMESTAMP, TIMESTAMP) + '.json', 'w', "utf-8") as json_file:
                json.dump({k: v for k, v in config._get_kwargs()}, json_file, ensure_ascii=False)

            with open(os.path.join(self.model_path, TIMESTAMP, TIMESTAMP) + '.pkl', 'wb') as f:
                pickle.dump({'seed': self.seed}, f, -1)

            for epoch in range(1, self.epoch + 1):
                epoch += self.epoch * fold_index
                epoch_loss, num_correct, num_pred = 0, 0, 0

                tbar = tqdm(train_loader)
                for i, (images, labels) in enumerate(tbar):
                    encoded, decoded = self.model(images)
                    loss = self.criterion(decoded, images.to(self.device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                    self.writer.add_scalar('train_loss', loss.item(), global_step + i)
                    params_groups_lr = str()
                    for group_ind, param_group in enumerate(optimizer.param_groups):
                        params_groups_lr = params_groups_lr + 'params_group_%d' % group_ind + ': %.12f, ' % (
                            param_group['lr'])
                    descript = "Fold: %d, Train Loss: %.7f, lr: %s" % (fold_index, loss.item(), params_groups_lr)
                    tbar.set_description(desc=descript)

                lr_scheduler.step()
                global_step += len(train_loader)
                val_loss = self.validation(val_loader)
                print('Finish Epoch [%d/%d] | Average training Loss: %.7f | Average validation Loss: %.7f |' % (
                    epoch, self.epoch * config.n_splits, epoch_loss / len(tbar), val_loss))

                if val_loss <= self.min_loss:
                    is_best = True
                    self.min_loss = val_loss
                else:
                    is_best = False

                state = {
                    'epoch': epoch,
                    'state_dict': self.model.module.state_dict(),
                    'min_loss': self.min_loss,
                }

                save_path = os.path.join(self.model_path, TIMESTAMP, TIMESTAMP + '.pth')
                torch.save(state, save_path)
                if is_best:
                    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Saving Best Model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    save_best_path = save_path.replace('.pth', '_best_{:.7f}.pth'.format(val_loss))
                    shutil.copyfile(save_path, save_best_path)
                self.writer.add_scalar('valid_loss', val_loss, epoch)

    def validation(self, valid_loader):
        self.model.eval()
        tbar = tqdm(valid_loader)
        loss_sum = 0

        with torch.no_grad():
            for i, (images, labels) in enumerate(tbar):
                _, decoded = self.model(images)
                loss = self.criterion(decoded, images.to(self.device))
                loss_sum += loss.item()

                descript = "Val Loss: {:.7f}".format(loss.item())
                tbar.set_description(desc=descript)
        loss_mean = loss_sum / len(tbar)

        return loss_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=50, help='epoch')
    parser.add_argument('--n_splits', type=int, default=10, help='n_splits_fold')
    parser.add_argument("--device", type=int, nargs='+', default=[i for i in range(torch.cuda.device_count())])
    parser.add_argument('--create_data', type=bool, default=False, help='For the first training')
    # model set
    parser.add_argument('--model_name', type=str, default='unet',
                        help='unet_resnet34/unet_se_resnext50_32x4d/unet_efficientnet_b4'
                             '/unet_resnet50/unet_efficientnet_b4')
    # model hyper-parameters
    parser.add_argument('--class_num', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4, help='init lr')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay in optimizer')
    # dataset
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--root', type=str, default='/home/Yuanbincheng/data/Ni-tension test-electrolysis-1-0.01-AE-20201031')
    parser.add_argument('--load_path', type=str, default='')
    config = parser.parse_args()
    print(config)

    train_val = TrainVal(config)
    train_val.train(config.create_data)
