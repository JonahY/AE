import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm
import csv
from plot_format import plot_norm
from collections import Counter


class Features:
    def __init__(self, color_1, color_2, time, feature_idx, interval_num=6):
        self.color_1 = color_1
        self.color_2 = color_2
        self.time = time
        self.interval_num = interval_num
        self.interval = 1 / interval_num
        self.feature_idx = feature_idx

    def cal_interval(self):
        interz, midz = [], []
        for tmp in self.feature_idx:
            # tmp = chan_2[:, idx]
            tmp_max = int(max(tmp))
            tmp_min = int(min(tmp))
            if tmp_min <= 0:
                interz.append([0] + [pow(10, i) for i in range(len(str(tmp_max)))])
                midz.append([self.interval * pow(10, i)
                             for i in range(len(str(tmp_max)) + 1)])
            else:
                interz.append([pow(10, i) for i in range(len(str(tmp_min)) - 1,
                                                         len(str(tmp_max)))])
                midz.append([self.interval * pow(10, i)
                             for i in range(len(str(tmp_min)),
                                            len(str(tmp_max)) + 1)])
        return interz, midz

    def cal_waitingTime_interval(self, res):
        tmp = sorted(np.array(res))
        tmp_min, tmp_max = math.floor(np.log10(min(tmp))), math.ceil(np.log10(max(tmp)))
        inter = [pow(10, i) for i in range(tmp_min, tmp_max + 1)]
        mid = [self.interval * pow(10, i) for i in range(tmp_min + 1, tmp_max + 2)]
        return inter, mid

    def cal_linear(self, tmp, inter, mid, idx=0):
        # 初始化横坐标
        x = np.array([])
        for i in inter:
            if i != 0:
                x = np.append(x, np.linspace(i, i * 10, self.interval_num, endpoint=False))
            else:
                x = np.append(x, np.linspace(i, 1, self.interval_num, endpoint=False))

        # 初始化纵坐标
        y = np.zeros(x.shape[0])
        for i, n in Counter(tmp).items():
            while True:
                try:
                    if x[idx] <= i < x[idx + 1]:
                        y[idx] += n
                        break
                except IndexError:
                    if x[idx] <= i:
                        y[idx] += n
                        break
                idx += 1

        # 对横坐标作进一步筛选，计算概率分布值
        x, y = x[y != 0], y[y != 0]
        xx = np.zeros(x.shape[0])
        yy = y / sum(y)

        # 取区间终点作为该段的横坐标
        for idx in range(len(x) - 1):
            xx[idx] = (x[idx] + x[idx + 1]) / 2
        xx[-1] = x[-1]

        # 计算分段区间长度，从而求得概率密度值
        interval = []
        for i, j in enumerate(mid):
            try:
                num = len(np.intersect1d(np.where(inter[i] <= xx)[0],
                                         np.where(xx < inter[i + 1])[0]))
                interval.extend([j] * num)
            except IndexError:
                num = len(np.where(inter[i] <= xx)[0])
                interval.extend([j] * num)
        yy = yy / np.array(interval)
        #     # 取对数变换为线性关系
        #     log_xx = np.log10(xx)
        #     log_yy = np.log10(yy)
        #     fit = np.polyfit(log_xx, log_yy, 1)
        #     alpha = abs(fit[0])
        #     fit_x = np.linspace(min(log_xx), max(log_xx), 100)
        #     fit_y = np.polyval(fit, fit_x)
        return xx, yy

    def cal_windows(self, tmp, samplerate=20000000):
        eny_lim = [[0.01, 0.1], [0.1, 1], [1, 10], [10, 100]]
        res = [[], [], [], []]
        init = []
        freq = 1 / samplerate

        for idx in tqdm(range(len(eny_lim))):
            i = 0
            while i < tmp.shape[0]:
                if eny_lim[idx][0] < tmp[i] < eny_lim[idx][1]:
                    if not init:
                        init.append(i)
                    else:
                        j = init.pop()
                        if i > j + 1:
                            k = (np.argmax(tmp[j + 1:i]) + 1) * freq
                            res[idx].append(k)
                elif tmp[i] > eny_lim[idx][1]:
                    if init:
                        j = init.pop()
                        if i > j + 1:
                            k = (np.argmax(tmp[j + 1:i]) + 1) * freq
                            res[idx].append(k)
                i += 1
        return res

    def find_max(self, res):
        lenz = []
        for i in range(len(res)):
            lenz.append(len(res[i]))
        idx = np.argmax(np.array(lenz))
        return idx

    def cal_PDF(self, tmp_origin, features_path, inter, mid, tmp_1, tmp_2, xlabel, ylabel):
        xx_origin, yy_origin = self.cal_linear(tmp_origin, inter, mid)
        xx_1, yy_1 = self.cal_linear(tmp_1, inter, mid)
        xx_2, yy_2 = self.cal_linear(tmp_2, inter, mid)

        for idx, [xx, yy] in enumerate(zip([xx_origin, xx_1, xx_2], [yy_origin, yy_1, yy_2])):
            with open(features_path[:-4] + '_%d ' % (idx) + ylabel + '.txt', 'w') as f:
                f.write('{}, {}\n'.format(xlabel, ylabel))
                for j in range(len(xx)):
                    f.write('{}, {}\n'.format(xx[j], yy[j]))

        fig = plt.figure(figsize=[6, 4.5], num='PDF--%s' % xlabel)
        ax = plt.subplot()
        for [xx, yy, color, label] in zip([xx_origin, xx_1, xx_2], [yy_origin, yy_1, yy_2],
                                          ['y', self.color_1, self.color_2], ['Origin', 'Class 1', 'Class 2']):
            ax.scatter(np.log10(xx), np.log10(yy), s=25, c=color, label=label)
        plot_norm(ax, xlabel, ylabel, legend_loc='upper right')

    def cal_CCDF(self, tmp_origin, features_path, tmp_1, tmp_2, xlabel, ylabel):
        N_origin, N1, N2 = len(tmp_origin), len(tmp_1), len(tmp_2)
        xx_origin, xx_1, xx_2 = [], [], []
        yy_origin, yy_1, yy_2 = [], [], []

        for idx, [tmp, N, xx, yy] in enumerate(zip([tmp_origin, tmp_1, tmp_2], [N_origin, N1, N2],
                                                   [xx_origin, xx_1, xx_2], [yy_origin, yy_1, yy_2])):
            for i in range(N - 1):
                xx.append(np.mean([tmp[i], tmp[i + 1]]))
                yy.append((N - i + 1) / N)

        for idx, [xx, yy] in enumerate(zip([xx_origin, xx_1, xx_2], [yy_origin, yy_1, yy_2])):
            with open(features_path[:-4] + '_%d ' % (idx) + 'CCDF(%s).txt' % xlabel[0], 'w') as f:
                f.write('{}, {}\n'.format(xlabel, ylabel))
                for j in range(len(xx)):
                    f.write('{}, {}\n'.format(xx[j], yy[j]))

        fig = plt.figure(figsize=[6, 4.5], num='CCDF--%s' % xlabel)
        ax = plt.subplot()
        for [xx, yy, color, label] in zip([xx_origin, xx_1, xx_2], [yy_origin, yy_1, yy_2],
                                          ['y', self.color_1, self.color_2], ['Origin', 'Class 1', 'Class 2']):
            ax.plot(np.log10(xx), np.log10(yy), markersize=25, color=color, label=label)
        plot_norm(ax, xlabel, ylabel, legend_loc='upper right')

    def cal_ML(self, tmp_origin, features_path, tmp_1, tmp_2, xlabel, ylabel):
        N_origin, N1, N2 = len(tmp_origin), len(tmp_1), len(tmp_2)
        ML_y_origin, ML_y1, ML_y2 = [], [], []
        Error_bar_origin, Error_bar1, Error_bar2 = [], [], []

        for idx, [tmp, N, ML_y, Error_bar] in enumerate(zip([tmp_origin, tmp_1, tmp_2], [N_origin, N1, N2],
                                                            [ML_y_origin, ML_y1, ML_y2],
                                                            [Error_bar_origin, Error_bar1, Error_bar2])):
            for j in tqdm(range(N)):
                valid_x = sorted(tmp)[j:]
                E0 = valid_x[0]
                Sum = np.sum(np.log(valid_x / E0))
                N_prime = N - j
                alpha = 1 + N_prime / Sum
                error_bar = (alpha - 1) / pow(N_prime, 0.5)
                ML_y.append(alpha)
                Error_bar.append(error_bar)

        for idx, [tmp, ML_y, Error_bar] in enumerate(zip([tmp_origin, tmp_1, tmp_2], [ML_y_origin, ML_y1, ML_y2],
                                                         [Error_bar_origin, Error_bar1, Error_bar2])):
            with open(features_path[:-4] + '_%d ' % (idx) + 'ML(%s).txt' % xlabel[0], 'w') as f:
                f.write('{}, {}, Error bar\n'.format(xlabel, ylabel))
                for j in range(len(ML_y)):
                    f.write('{}, {}, {}\n'.format(sorted(tmp)[j], ML_y[j], Error_bar[j]))

        fig = plt.figure(figsize=[6, 4.5], num='ML--%s' % xlabel)
        ax = plt.subplot()
        for [tmp, ML_y, Error_bar, color, label] in zip([tmp_origin, tmp_1, tmp_2], [ML_y_origin, ML_y1, ML_y2],
                                                        [Error_bar_origin, Error_bar1, Error_bar2],
                                                        ['y', self.color_1, self.color_2],
                                                        ['Origin', 'Class 1', 'Class 2']):
            ax.errorbar(np.log10(sorted(tmp)), ML_y, yerr=Error_bar, fmt='o', ecolor=color,
                        color=color, elinewidth=1, capsize=2, ms=5, label=label)
        plot_norm(ax, xlabel, ylabel, y_lim=[1.25, 3])

    def cal_contour(self, tmp_1, tmp_2, xlabel, ylabel, title, padding=False, clabel=False):
        tmp_1, tmp_2 = 20 * np.log10(tmp_1), 20 * np.log10(tmp_2)
        x, y = np.linspace(0, 100, 75), np.linspace(-20, 80, 65)
        X, Y = np.meshgrid(x, y)
        height = np.zeros([X.shape[0], Y.shape[1]])
        linestyles = ['solid'] * 6 + ['--'] * 4
        levels = [0.5, 1.5, 3, 6, 12, 24, 48, 96, 192, 384]
        colors = ['navy', 'blueviolet', 'cornflowerblue', 'lightsteelblue',
                  'aliceblue', 'lightyellow', 'gold', 'yellow', 'red', 'darkred']

        for i in range(X.shape[1] - 1):
            valid_x = np.where((tmp_1 < X[0, i + 1]) & (tmp_1 >= X[0, i]))[0]
            for j in range(Y.shape[0] - 1):
                valid_y = np.where((tmp_2 < Y[j + 1, 0]) & (tmp_2 >= Y[j, 0]))[0]

                height[j, i] = np.intersect1d(valid_x, valid_y).shape[0]

        fig = plt.figure(figsize=[6, 4.5], num='Contour--%s & %s' % (ylabel, xlabel))
        ax = plt.subplot()
        if padding:
            ctf = ax.contourf(X, Y, height, levels, colors=colors, extend='max')
            cbar = plt.colorbar(ctf)
        else:
            ct = plt.contour(X, Y, height, levels, colors=colors, linewidths=1, linestyles=linestyles)
            cbar = plt.colorbar(ct)
        if clabel:
            ax.clabel(ct, inline=True, colors='k', fmt='%.1f')
        plot_norm(ax, xlabel, ylabel, title, legend=False)

    def plot_correlation(self, tmp_1, tmp_2, xlabel, ylabel, chan, cls_1=None, cls_2=None, idx_1=None, idx_2=None):
        fig = plt.figure(figsize=[6, 4.5], num='Correlation--%s & %s' % (ylabel, xlabel))
        ax = plt.subplot()
        if cls_1 is not None and cls_2 is not None:
            ax.scatter(np.log10(tmp_1)[cls_2], np.log10(tmp_2)[cls_2], s=25, c=self.color_2, label='Class 2')
            ax.scatter(np.log10(tmp_1)[cls_1], np.log10(tmp_2)[cls_1], s=25, c=self.color_1, label='Class 1')
            if idx_1:
                ax.scatter(np.log10(tmp_1)[cls_1][idx_1], np.log10(tmp_2)[cls_1][idx_1], s=25, c='y',
                           label='Random selection')
            if idx_2:
                ax.scatter(np.log10(tmp_1)[cls_2][idx_2], np.log10(tmp_2)[cls_2][idx_2], s=25, c='purple',
                           label='Random selection')
        else:
            ax.scatter(np.log10(tmp_1), np.log10(tmp_2), s=25, c='g')
            plot_norm(ax, xlabel, ylabel, chan, legend=False)
        plot_norm(ax, xlabel, ylabel, chan)

    def plot_feature_time(self, tmp, ylabel):
        fig = plt.figure(figsize=[6, 4.5], num='Time domain curve')
        ax = plt.subplot()
        ax.scatter(self.time, np.log10(tmp))
        ax.set_xticks(np.linspace(0, 40000, 9))
        ax.set_yticks([-1, 0, 1, 2, 3])
        plot_norm(ax, 'Time(s)', ylabel, legend=False, formatter_y=True)

    def cal_waitingTime(self, tmp_origin, features_path, cls_1, cls_2, xlabel, ylabel):
        tmp_origin, tmp_1, tmp_2 = self.cal_windows(tmp_origin), self.cal_windows(tmp_origin[cls_1]), \
                                   self.cal_windows(tmp_origin[cls_2])

        fig = plt.figure(figsize=[6, 4.5], num='Waiting Time Curve')
        ax = plt.subplot()
        for idx, [tmp, color, label] in enumerate(
                zip([tmp_origin, tmp_1, tmp_2], ['y', self.color_1, self.color_2], ['Origin', 'Class 1', 'Class 2'])):
            i = self.find_max(tmp)
            inter, mid = self.cal_waitingTime_interval(tmp[i])
            xx, yy = self.cal_linear(sorted(np.array(tmp[i])), inter, mid)
            ax.scatter(np.log10(xx), np.log10(yy), s=25, c=color, label=label)
            with open(features_path[:-4] + '_pop%d ' % (idx) + ylabel + '.txt', 'w') as f:
                f.write('{}, {}\n'.format(xlabel, ylabel))
                for j in range(len(xx)):
                    f.write('{}, {}\n'.format(xx[j], yy[j]))
        plot_norm(ax, xlabel, ylabel, legend_loc='upper right')


if __name__ == "__main__":
    path = r'C:\Users\Yuan\Desktop'
    # E:\data\vallen
    # E:\data\CM-PM-o18-2020.10.17
    # E:\data\CM-4M-o18-2020.10.17-1-60
    # C:\Users\Yuan\Desktop
    os.chdir(path)
    features_path = r'pri_database.txt'
    # Ni-tension test-electrolysis-1-0.01-AE-20201031
    # r'C:\Users\Yuan\Desktop\pri_database.txt'
    # r'C:\Users\Yuan\Desktop\CM-4M-o18-2020.10.17-1-60.txt'
    # r'E:\data\CM-PM-o18-2020.10.17\CM-PM-o18-2020.10.17.txt'

    label_path = r'C:\Users\Yuan\Desktop\label.txt'

    # Time,Chan,Status,Thr,Amp,RiseT,Dur,Eny,RMS,Counts,TRAI
    with open(features_path, 'r') as f:
        feature = np.array([i.split(',')[2:-3] for i in f.readlines()[1:]])
    feature = feature.astype(np.float32)

    with open(label_path, 'r') as f:
        label = np.array([i.strip() for i in f.readlines()[1:]])
    label = label.astype(np.float32).reshape(-1, 1)
    label[np.where(label == 2)] = 0
    ext = np.zeros([feature.shape[0], 1])
    ext[np.where(label == 0)[0].tolist()] = 1
    label = np.concatenate((label, ext), axis=1)
    cls_1 = label[:, 0] == 1
    cls_2 = label[:, 1] == 1

    feature_idx = [4, 6, 7]
    xlabelz = ['Amplitude(μV)', 'Duration(μs)', 'Energy(aJ)']
    ylabelz = ['PDF(A)', 'PDF(D)', 'PDF(E)']
    color_1 = [255 / 255, 0 / 255, 102 / 255]  # red
    color_2 = [0 / 255, 136 / 255, 204 / 255]  # blue
    features = Features(color_1, color_2, feature[:, 0])
    interz, midz = features.cal_interval()

    for i, [idx, inter, mid, xlabel, ylabel] in enumerate(zip(feature_idx, interz, midz, xlabelz, ylabelz)):
        # tmp = feature[:, idx] * pow(10, 6) if idx == 3 else feature[:, idx]
        tmp = sorted(feature[:, idx])
        tmp_1, tmp_2 = tmp[cls_1], tmp[cls_2]
        features.cal_PDF(tmp, features_path, interz[i], midz[i], interval_num, tmp_1, tmp_2, xlabel, ylabel)
        features.cal_CCDF(tmp, features_path, tmp_1, tmp_2, xlabel, 'CCD C(s)')
        features.cal_ML(tmp, features_path, tmp_1, tmp_2, xlabel, r'$\epsilon$')

    features.cal_contour(feature[:, 7], feature[:, 6], 'Energy(aJ)', 'Duration(μs)', 'Contour')
    features.plot_correlation(feature[:, 6], feature[:, 7], 'Duration(μs)', 'Energy(aJ)', 'Chan 2', cls_1, cls_2)
    plt.show()
