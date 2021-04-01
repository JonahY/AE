from torch import optim
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from solver import Solver
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
import datetime
import os
import codecs, json
import time
import argparse
import numpy as np
import pandas as pd
import shutil

from meter import Meter
from set_seed import seed_torch
import pickle
import random
from network import ClassifyResNet
from dataset import classify_provider
from loss import ClassifyLoss

from torchvision import models
from alexnet_pytorch import AlexNet
from ResNeXt import ResNeXt
# from Alexnet import AlexNet
# from VGG import *
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class TrainVal():
    def __init__(self, config):
        self.model = models.shufflenet_v2_x1_0(pretrained=True)

        # # freeze model parameters
        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.model.fc = nn.Sequential(nn.Linear(1024, config.class_num), nn.Sigmoid())
        # for param in self.model.fc.parameters():
        #     param.requires_grad = True

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

        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.epoch = config.epoch
        self.splits = config.n_splits
        self.root = config.root

        self.solver = Solver(self.model, self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.TIME = "{0:%Y-%m-%dT%H-%M-%S}-classify".format(datetime.datetime.now())
        self.model_path = os.path.join(config.root, config.save_path, config.model_name, self.TIME)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.max_accuracy_valid = 0
        self.seed = int(time.time())
        # self.seed = 1570421136
        seed_torch(self.seed)

        self.train_transform = transforms.Compose(
            [transforms.Resize([256, 256]),
             transforms.RandomCrop(224),
             transforms.RandomRotation(degrees=(-40, 40)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()])
        self.test_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    def train(self, create_data=False):
        if create_data:
            df = pd.read_csv(os.path.join(self.root, 'train info_cwt_coarse.csv'), header=None)
            labels_1dim = np.argmax(np.array(df), axis=1)
            print('<' * 20 + ' Start creating datasets ' + '>' * 20)
            skf = StratifiedKFold(n_splits=self.splits, shuffle=True, random_state=55)
            for idx, [train_df_index, val_df_index] in tqdm(enumerate(skf.split(df, labels_1dim), 1)):
                for i in train_df_index:
                    try:
                        shutil.copy(os.path.join(config.root, 'Ni-coarse-cwt/%s.jpg' % (i + 1)),
                                    os.path.join('/home/Yuanbincheng/project/dislocation_cls/2/4cls',
                                                 'train_%d/%d/%d.jpg' % (idx, labels_1dim[i], i + 1)))
                    except FileNotFoundError:
                        try:
                            os.mkdir(os.path.join('/home/Yuanbincheng/project/dislocation_cls/2/4cls', 'train_%d' % idx))
                        except (FileNotFoundError, FileExistsError):
                            os.mkdir(os.path.join('/home/Yuanbincheng/project/dislocation_cls/2/4cls',
                                                  'train_%d/%d' % (idx, labels_1dim[i])))
                for i in val_df_index:
                    try:
                        shutil.copy(os.path.join(config.root, 'Ni-coarse-cwt/%s.jpg' % (i + 1)),
                                    os.path.join('/home/Yuanbincheng/project/dislocation_cls/2/4cls',
                                                 'test_%d/%d/%d.jpg' % (idx, labels_1dim[i], i + 1)))
                    except FileNotFoundError:
                        try:
                            os.mkdir(os.path.join('/home/Yuanbincheng/project/dislocation_cls/2/4cls', 'test_%d' % idx))
                        except (FileNotFoundError, FileExistsError):
                            os.mkdir(os.path.join('/home/Yuanbincheng/project/dislocation_cls/2/4cls',
                                                  'test_%d/%d' % (idx, labels_1dim[i])))
            print('<' * 20 + ' Finish creating datasets ' + '>' * 20)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.module.parameters()), self.lr,
                               weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.99)
        global_step, global_threshold, global_threshold_pop1, global_threshold_pop2, global_threshold_pop3 = 1, 1, 1, 1, 1

        for fold_index in range(self.splits):
            train_dataset = torchvision.datasets.ImageFolder(root='4cls/train_%d/' % (fold_index + 1), transform=self.train_transform)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
            val_dataset = torchvision.datasets.ImageFolder(root='4cls/test_%d/' % (fold_index + 1), transform=self.test_transform)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
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
                    labels_predict = self.solver.forward(images)
                    loss = self.solver.cal_loss(labels, labels_predict, self.criterion)
                    epoch_loss += loss.item()
                    self.solver.backword(optimizer, loss)

                    # tmp = (labels_predict > 0.2).float()

                    labels_predictIdx, labels_predictMax = torch.max(labels_predict, 1)[1].cpu(), torch.max(labels_predict, 1)[0].cpu()
                    correct_idx = labels_predictIdx == labels
                    num_correct += correct_idx.sum().item()
                    num_pred += labels_predictIdx.size(0)
                    # for p, t in zip(labels_predictMax[correct_idx], labels[correct_idx]):
                    for p in labels_predict.cpu()[correct_idx]:
                        self.writer.add_scalar('threshold_pop1', p[0].item(), global_threshold)
                        self.writer.add_scalar('threshold_pop2', p[1].item(), global_threshold)
                        self.writer.add_scalar('threshold_pop3', p[2].item(), global_threshold)
                        self.writer.add_scalar('threshold_pop4', p[3].item(), global_threshold)
                        global_threshold += 1
                        # if t == 0:
                        #     self.writer.add_scalar('threshold_pop1', p.item(), global_threshold_pop1)
                        #     global_threshold_pop1 += 1
                        # elif t == 1:
                        #     self.writer.add_scalar('threshold_pop2', p.item(), global_threshold_pop2)
                        #     global_threshold_pop2 += 1
                        # elif t == 2:
                        #     self.writer.add_scalar('threshold_pop3', p.item(), global_threshold_pop3)
                        #     global_threshold_pop3 += 1

                    self.writer.add_scalar('train_loss', loss.item(), global_step + i)
                    params_groups_lr = str()
                    for group_ind, param_group in enumerate(optimizer.param_groups):
                        params_groups_lr = params_groups_lr + 'params_group_%d' % (group_ind) + ': %.12f, ' % (
                        param_group['lr'])
                    descript = "Fold: %d, Train Loss: %.7f, lr: %s" % (fold_index, loss.item(), params_groups_lr)
                    tbar.set_description(desc=descript)

                lr_scheduler.step()
                global_step += len(train_loader)
                precision, recall, f1, val_loss, val_accuracy = self.validation(val_loader)
                print('Finish Epoch [%d/%d] | Average training Loss: %.7f | Training accuracy: %.4f | Average validation Loss: %.7f | Validation accuracy: %.4f |' % (
                epoch, self.epoch * config.n_splits, epoch_loss / len(tbar), num_correct / num_pred, val_loss, val_accuracy))

                if val_accuracy > self.max_accuracy_valid:
                    is_best = True
                    self.max_accuracy_valid = val_accuracy
                else:
                    is_best = False

                state = {
                    'epoch': epoch,
                    'state_dict': self.model.module.state_dict(),
                    'max_accuracy_valid': self.max_accuracy_valid,
                }

                self.solver.save_checkpoint(os.path.join(self.model_path, TIMESTAMP, TIMESTAMP + '.pth'), state,
                                            is_best, self.max_accuracy_valid)
                self.writer.add_scalar('valid_loss', val_loss, epoch)
                self.writer.add_scalar('valid_accuracy', val_accuracy, epoch)
                self.writer.add_scalar('valid_class_1_f1', f1[0], epoch)
                self.writer.add_scalar('valid_class_2_f1', f1[1], epoch)
                self.writer.add_scalar('valid_class_3_f1', f1[2], epoch)
                self.writer.add_scalar('valid_class_4_f1', f1[3], epoch)

    def validation(self, valid_loader):
        self.model.eval()
        tbar = tqdm(valid_loader)
        loss_sum, num_correct, num_pred = 0, 0, 0
        y_true, y_pre = [], []

        with torch.no_grad():
            for i, (images, labels) in enumerate(tbar):
                labels_predict = self.solver.forward(images)
                loss = self.solver.cal_loss(labels, labels_predict, self.criterion)
                loss_sum += loss.item()

                # tmp = (labels_predict > 0.2).float()
                labels_predictIdx = torch.max(labels_predict, 1)[1].cpu()
                num_correct += (labels_predictIdx == labels).sum().item()
                num_pred += labels_predictIdx.size(0)
                y_true.extend(labels.numpy().tolist())
                y_pre.extend(labels_predictIdx.numpy().tolist())

                descript = "Val Loss: {:.7f}".format(loss.item())
                tbar.set_description(desc=descript)
        loss_mean = loss_sum / len(tbar)
        res = confusion_matrix(y_true, y_pre)
        precision = np.array([res[i][i] / np.sum(res, axis=0)[i] for i in range(config.class_num)])
        recall = np.array([res[i][i] / np.sum(res, axis=1)[i] for i in range(config.class_num)])
        f1 = 2 * precision * recall / (precision + recall)
        for idx, [p, r, f] in enumerate(zip(precision, recall, f1)):
            print("Class_%d_precision: %0.4f | Recall: %0.4f | F1-score: %0.4f |" % (idx, p, r, f))
        return precision, recall, f1, loss_mean, num_correct / num_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epoch', type=int, default=50, help='epoch')
    parser.add_argument('--n_splits', type=int, default=10, help='n_splits_fold')
    parser.add_argument("--device", type=int, nargs='+', default=[i for i in range(torch.cuda.device_count())])
    parser.add_argument('--create_data', type=bool, default=False, help='For the first training')
    # model set
    parser.add_argument('--model_name', type=str, default='shufflenet',
                        help='unet_resnet34/unet_se_resnext50_32x4d/unet_efficientnet_b4'
                             '/unet_resnet50/unet_efficientnet_b4')
    # model hyper-parameters
    parser.add_argument('--class_num', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4, help='init lr')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay in optimizer')
    # dataset
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--root', type=str, default='/home/Yuanbincheng/data/Ni-tension test-pure-1-0.01-AE-20201030')
    config = parser.parse_args()
    print(config)

    train_val = TrainVal(config)
    train_val.train(config.create_data)
