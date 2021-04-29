import os
import cv2
import warnings
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import sys
from tqdm import tqdm
from PIL import Image
from utils import data_augmentation, image_torch

sys.path.append('.')
warnings.filterwarnings("ignore")


def classify_provider(root, fold, n_splits, batch_size, num_workers, resize, mean, std, crop, height, width):
    df = pd.read_csv(os.path.join(root, 'train info_wsst.csv'), header=None)
    labels_1dim = np.argmax(np.array(df), axis=1)
    print('<' * 20 + ' Start creating datasets ' + '>' * 20)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=55)
    for idx, [train_df_index, val_df_index] in tqdm(enumerate(skf.split(df, labels_1dim), 1)):
        for i in train_df_index:
            try:
                shutil.copy(os.path.join(config.root, 'train dataset_wsst/%s.jpg' % (i + 1)),
                            os.path.join('/home/Yuanbincheng/project/dislocation_cls/3/SAE_316',
                                         'train_%d/%d/%d.jpg' % (idx, labels_1dim[i], i + 1)))
            except FileNotFoundError:
                try:
                    os.mkdir(
                        os.path.join('/home/Yuanbincheng/project/dislocation_cls/3/SAE_316', 'train_%d' % idx))
                except (FileNotFoundError, FileExistsError):
                    os.mkdir(os.path.join('/home/Yuanbincheng/project/dislocation_cls/3/SAE_316',
                                          'train_%d/%d' % (idx, labels_1dim[i])))
        for i in val_df_index:
            try:
                shutil.copy(os.path.join(config.root, 'train dataset_wsst/%s.jpg' % (i + 1)),
                            os.path.join('/home/Yuanbincheng/project/dislocation_cls/3/SAE_316',
                                         'test_%d/%d/%d.jpg' % (idx, labels_1dim[i], i + 1)))
            except FileNotFoundError:
                try:
                    os.mkdir(os.path.join('/home/Yuanbincheng/project/dislocation_cls/3/SAE_316', 'test_%d' % idx))
                except (FileNotFoundError, FileExistsError):
                    os.mkdir(os.path.join('/home/Yuanbincheng/project/dislocation_cls/3/SAE_316',
                                          'test_%d/%d' % (idx, labels_1dim[i])))
    return dataloaders


class AEClassDataset(Dataset):
    def __init__(self, df, data_folder, resize, mean, std, phase, crop=False, height=None, width=None):
        super(AEClassDataset, self).__init__()
        self.df = df
        self.root = data_folder
        self.resize = resize
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms
        self.crop = crop
        self.height = height
        self.width = width
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        label = np.argmax(np.array(self.df.iloc[idx, :]))
        image_path = os.path.join(self.root, "train dataset_cwt", str(idx + 1) + '.jpg')
        img = cv2.imread(image_path)
        img = self.transforms(self.phase, img, self.resize, self.mean, self.std, crop=self.crop, crop_height=self.height,
                              crop_width=self.width)
        return img, label

    def __len__(self):
        return len(self.fnames)


def augmentation(phase, image, resize, crop_height=None, crop_width=None):
    image_aug = data_augmentation(phase, image, resize, crop_height, crop_width)
    # image_aug = Image.fromarray(image_aug)

    return image_aug


def get_transforms(phase, image, resize, mean, std, crop=False, crop_height=None, crop_width=None):
    image = augmentation(phase, image, resize, crop_height, crop_width)
    # if phase == 'train':
    #     image = augmentation(phase, image, crop=crop, height=height, width=width)
    #     normalize = transforms.Normalize(mean, std)
    #     transform_compose = transforms.Compose([to_tensor])
    #     image = transform_compose(image)

    return image


if __name__ == "__main__":
    data_folder = "/home/Yuanbincheng/data/Ni-tension test-pure-1-0.01-AE-20201030"
    df_path = "train info.csv"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size = 12
    num_workers = 1
    n_splits = 5
    mask_only = False
    crop = False
    height = 256
    width = 512
    # 测试分割数据集
    class_dataloader = classify_provider(data_folder, df_path, n_splits, batch_size, num_workers, mean, std, crop, height, width)
    for fold_index, [classify_train_dataloader, classify_val_dataloader] in enumerate(class_dataloader):
        class_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        for [classify_images, classify_targets] in tqdm(classify_val_dataloader):
            image = classify_images[0]
            target = classify_targets[0]
            image = image_torch(image, mean, std)
            classify_target = classify_targets[0]
            position_x = 10
            for i in range(classify_target.size(0)):
                color = class_color[i]
                position_x += 50
                position = (position_x, 50)
                if classify_target[i] != 0:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    img_ = image.copy()
                    image = cv2.putText(img_, str(i+1), position, font, 1.2, color, 2)
            cv2.imshow('win', image)
            cv2.waitKey(0)
    pass
