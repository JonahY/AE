# -*- coding: UTF-8 -*-
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
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm
import array
import csv
import sqlite3
from kmeans import KernelKMeans, ICA
from utils import *
import sys


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

    def cal_negtive_interval(self, res):
        tmp = sorted(np.array(res))
        tmp_min, tmp_max = math.floor(np.log10(min(tmp))), math.ceil(np.log10(max(tmp)))
        inter = [pow(10, i) for i in range(tmp_min, tmp_max+1)]
        mid = [interval * pow(10, i) for i in range(tmp_min+1, tmp_max+2)]
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

    def cal_N_Naft(self, tmp, eny_lim):
        N_ms, N_as = 0, 0
        main_peak = np.where(eny_lim[0] < tmp)[0]
        if len(main_peak):
            for i in range(main_peak.shape[0] - 1):
                if main_peak[i] >= eny_lim[1]:
                    continue
                elif main_peak[i+1] - main_peak[i] == 1:
                    N_ms += tmp[main_peak[i]]
                    continue
                N_ms += tmp[main_peak[i]]
                N_as += np.max(tmp[main_peak[i]+1:main_peak[i+1]])
            if main_peak[-1] < tmp.shape[0] - 1:
                N_as += np.max(tmp[main_peak[-1]+1:])
            N_ms += tmp[main_peak[-1]]
        return N_ms + N_as, N_as

    def cal_OmiroLaw_helper(self, tmp, eny_lim):
        res = [[] for _ in range(len(eny_lim))]
        for idx in tqdm(range(len(eny_lim))):
            main_peak = np.where((eny_lim[idx][0] < tmp) & (tmp < eny_lim[idx][1]))[0]
            if len(main_peak):
                for i in range(main_peak.shape[0] - 1):
                    for j in range(main_peak[i]+1, main_peak[i+1]+1):
                        if tmp[j] < eny_lim[idx][1]:
                            k = Time[j] - Time[main_peak[i]]
                            res[idx].append(k)
                        else:
                            break
                if main_peak[-1] < tmp.shape[0] - 1:
                    for j in range(main_peak[-1] + 1, tmp.shape[0]):
                        k = Time[j] - Time[main_peak[-1]]
                        res[idx].append(k)
        return res

    def cal_PDF(self, tmp_origin, features_path, inter, mid, tmp_1, tmp_2, xlabel, ylabel):
        fig = plt.figure(figsize=[6, 3.9], num='PDF--%s' % xlabel)
        ax = plt.subplot()
        for tmp, color, label in zip([tmp_origin, tmp_1, tmp_2], ['black', self.color_1, self.color_2],
                                     ['whole', 'population 1', 'population 2']):
            xx, yy = self.cal_linear(tmp, inter, mid)
            ax.loglog(xx, yy, '.', Marker='.', markersize=8, color=color, label=label)
            with open(features_path[:-4] + '_{}_'.format(label) + ylabel + '.txt', 'w') as f:
                f.write('{}, {}\n'.format(xlabel, ylabel))
                for j in range(len(xx)):
                    f.write('{}, {}\n'.format(xx[j], yy[j]))
        plot_norm(ax, xlabel, ylabel, legend_loc='upper right')

    def cal_CCDF(self, tmp_origin, features_path, tmp_1, tmp_2, xlabel, ylabel):
        N_origin, N1, N2 = len(tmp_origin), len(tmp_1), len(tmp_2)

        fig = plt.figure(figsize=[6, 3.9], num='CCDF--%s' % xlabel)
        ax = plt.subplot()
        for tmp, N, color, label in zip([tmp_origin, tmp_1, tmp_2], [N_origin, N1, N2],
                                        ['black', self.color_1, self.color_2],
                                        ['whole', 'population 1', 'population 2']):
            xx, yy = [], []
            for i in range(N - 1):
                xx.append(np.mean([tmp[i], tmp[i + 1]]))
                yy.append((N - i + 1) / N)
            ax.loglog(xx, yy, color=color, label=label)
#             ax.plot(np.log10(xx), np.log10(yy), markersize=25, color=color, label=label)
            with open(features_path[:-4] + '_{}_'.format(label) + 'CCDF(%s).txt' % xlabel[0], 'w') as f:
                f.write('{}, {}\n'.format(xlabel, ylabel))
                for j in range(len(xx)):
                    f.write('{}, {}\n'.format(xx[j], yy[j]))
        plot_norm(ax, xlabel, ylabel, legend_loc='upper right')

    def cal_ML(self, tmp_origin, features_path, tmp_1, tmp_2, xlabel, ylabel):
        N_origin, N1, N2 = len(tmp_origin), len(tmp_1), len(tmp_2)

        fig = plt.figure(figsize=[6, 3.9], num='ML--%s' % xlabel)
        ax = plt.subplot()
        ax.set_xscale("log", nonposx='clip')
        for tmp, N, layer, color, label in zip([tmp_origin, tmp_1, tmp_2], [N_origin, N1, N2], [0.1, 0.2, 0.3],
                                               ['black', self.color_1, self.color_2],
                                               ['whole', 'population 1', 'population 2']):
            ML_y, Error_bar = [], []
            for j in tqdm(range(N)):
                valid_x = sorted(tmp)[j:]
                E0 = valid_x[0]
                Sum = np.sum(np.log(valid_x / E0))
                N_prime = N - j
                alpha = 1 + N_prime / Sum
                error_bar = (alpha - 1) / pow(N_prime, 0.5)
                ML_y.append(alpha)
                Error_bar.append(error_bar)
            ax.errorbar(sorted(tmp), ML_y, yerr=Error_bar, fmt='o', ecolor=color, color=color, elinewidth=1, capsize=2,
                        ms=3, label=label, zorder=layer)
            with open(features_path[:-4] + '_{}_'.format(label) + 'ML(%s).txt' % xlabel[0], 'w') as f:
                f.write('{}, {}, Error bar\n'.format(xlabel, ylabel))
                for j in range(len(ML_y)):
                    f.write('{}, {}, {}\n'.format(sorted(tmp)[j], ML_y[j], Error_bar[j]))
        plot_norm(ax, xlabel, ylabel, y_lim=[1.25, 3])

    def cal_contour(self, tmp_1, tmp_2, xlabel, ylabel, title, x_lim, y_lim, size_x=40, size_y=40,
                    method='linear_bin', padding=False, clabel=False):
        tmp_1, tmp_2 = 20 * np.log10(tmp_1), 20 * np.log10(tmp_2)
        if method == 'log_bin':
            sum_x, sum_y = x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]
            arry_x = np.logspace(np.log10(sum_x + 10), 1, size_x) / (sum(np.logspace(np.log10(sum_x + 10), 1, size_x)) / sum_x)
            arry_y = np.logspace(np.log10(sum_y + 10), 1, size_y) / (sum(np.logspace(np.log10(sum_y + 10), 1, size_y)) / sum_y)
            x, y = [], []
            for tmp, res, arry in zip([x_lim[0], y_lim[0]], [x, y], [arry_x, arry_y]):
                for i in arry:
                    res.append(tmp)
                    tmp += i
            x, y = np.array(x), np.array(y)
        elif method == 'linear_bin':
            x, y = np.linspace(x_lim[0], x_lim[1], size_x), np.linspace(y_lim[0], y_lim[1], size_y)
        X, Y = np.meshgrid(x, y)
        height = np.zeros([X.shape[0], Y.shape[1]])
        linestyles = ['solid'] * 8 + ['--'] * 4
        levels = [1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536]
        colors = [[1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0.5, 0.5, 0.5],
                  [1, 0.3, 0], [0, 0, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0]]

        for i in range(X.shape[1] - 1):
            valid_x = np.where((tmp_1 < X[0, i + 1]) & (tmp_1 >= X[0, i]))[0]
            for j in range(Y.shape[0] - 1):
                valid_y = np.where((tmp_2 < Y[j + 1, 0]) & (tmp_2 >= Y[j, 0]))[0]
                height[j, i] = np.intersect1d(valid_x, valid_y).shape[0]

        fig = plt.figure(figsize=[6, 3.9], num='Contour--%s & %s' % (ylabel.split(' ')[-1][0], xlabel.split(' ')[-1][0]))
        ax = plt.subplot()
        if padding:
            ctf = ax.contourf(X, Y, height, levels, colors=colors, extend='max')
            cbar = plt.colorbar(ctf)
        else:
            ct = plt.contour(X, Y, height, levels, colors=colors, linewidths=1, linestyles=linestyles)
            cbar = plt.colorbar(ct)
        if clabel:
            ax.clabel(ct, inline=True, colors='k', fmt='%.1f')
        plot_norm(ax, xlabel, ylabel, title=title, legend=False)

    def plot_correlation(self, tmp_1, tmp_2, xlabel, ylabel, cls_1=None, cls_2=None, idx_1=None, idx_2=None,
                         fit=False, status='A-D', x1_lim=None, x2_lim=None, plot_lim=None, title=''):
        fig = plt.figure(figsize=[6, 3.9], num='Correlation--%s & %s %s' % (ylabel, xlabel, title))
        ax = plt.subplot()
        if cls_1 is not None and cls_2 is not None:
            ax.loglog(tmp_1[cls_2], tmp_2[cls_2], '.', marker='.', markersize=8, color=self.color_2, label='Class 2')
            ax.loglog(tmp_1[cls_1], tmp_2[cls_1], '.', marker='.', markersize=8, color=self.color_1, label='Class 1')
            if idx_1:
                ax.loglog(tmp_1[cls_1][idx_1], tmp_2[cls_1][idx_1], '.', marker='.', markersize=8, color='black')
            if idx_2:
                ax.loglog(tmp_1[cls_2][idx_2], tmp_2[cls_2][idx_2], '.', marker='.', markersize=8, color='black')
        else:
            ax.loglog(tmp_1, tmp_2, '.', Marker='.', markersize=8, color='g')
            plot_norm(ax, xlabel, ylabel, legend=False)
        plot_norm(ax, xlabel, ylabel)

        if fit:
            cor_x1, cor_x2 = tmp_1[cls_1], tmp_1[cls_2]
            cor_y1, cor_y2 = tmp_2[cls_1], tmp_2[cls_2]
            if status == 'A-D':
                A = np.where((cor_x1 > x1_lim[0]) & (cor_x1 < x1_lim[1]))
                B = np.where((cor_x2 > x2_lim[0]) & (cor_x2 < x2_lim[1]))
            elif status == 'E-A':
                A = fit_with_x1
                B = fit_with_x2
            linear_x1 = cor_x1[A]
            linear_y1 = cor_y1[A]
            linear_x2 = cor_x2[B]
            linear_y2 = cor_y2[B]
            convert = lambda x, a, b: pow(x, a) * pow(10, b)
            ave = 0
            alpha, b, fit_x, fit_y = [], [], [], []
            mix_cor_x = [min(cor_x1), min(cor_x2)] if status == 'E-A' else plot_lim
            for linear_x, linear_y, min_x, max_x in zip([linear_x1, linear_x2], [linear_y1, linear_y2],
                                                        mix_cor_x, [max(cor_x1), max(cor_x2)]):
                fit = np.polyfit(np.log10(linear_x), np.log10(linear_y), 1)
                alpha.append(fit[0])
                b.append(fit[1])
                fit_x.append(np.linspace(min_x, max_x, 100))
                fit_y.append(convert(np.linspace(min_x, max_x, 100), fit[0], fit[1]))
            ax.plot(fit_x[0], fit_y[0], color='black')
            ax.plot(fit_x[1], fit_y[1], color='black')
            if status == 'A-D':
                min_y = max(min(fit_y[0]), min(fit_y[1]))
                max_y = min(max(fit_y[0]), max(fit_y[1]))
                cal_y = np.linspace(np.log10(min_y), np.log10(max_y), 100)
                for i in cal_y:
                    tmp1 = (i - b[0]) / alpha[0]
                    tmp2 = (i - b[1]) / alpha[1]
                    ave += max(pow(10, tmp1), pow(10, tmp2)) / min(pow(10, tmp1), pow(10, tmp2))
            elif status == 'E-A':
                min_x = max(min(fit_x[0]), min(fit_x[1]))
                max_x = min(max(fit_x[0]), max(fit_x[1]))
                cal_x = np.linspace(np.log10(min_x), np.log10(max_x), 100)
                for i in cal_x:
                    tmp1 = alpha[0] * i + b[0]
                    tmp2 = alpha[1] * i + b[1]
                    ave += max(pow(10, tmp1), pow(10, tmp2)) / min(pow(10, tmp1), pow(10, tmp2))
            return ave / 100, alpha, b, A, B

    def plot_feature_time(self, tmp, ylabel):
        fig = plt.figure(figsize=[6, 3.9], num='Time domain curve')
        ax = plt.subplot()
        ax.set_yscale("log", nonposy='clip')
        ax.scatter(self.time, tmp)
        ax.set_xticks(np.linspace(0, 40000, 9))
        ax.set_yticks([-1, 0, 1, 2, 3])
        plot_norm(ax, 'Time(s)', ylabel, legend=False)

    def cal_BathLaw(self, tmp_origin, tmp_1, tmp_2, xlabel, ylabel):
        fig = plt.figure(figsize=[6, 3.9], num='Bath law')
        ax = plt.subplot()
        for tmp, marker, color, label in zip([tmp_1, tmp_2], ['p', 'h'], [color_1, color_2], ['Population 1', 'Population 2']):
            tmp_max = int(max(tmp))
            inter = [pow(10, i) for i in range(0, len(str(tmp_max)))]
            x = np.array([])
            y = []
            for i in inter:
                x = np.append(x, np.linspace(i, i * 10, self.interval_num, endpoint=False))
            for k in range(x.shape[0]):
                if k != x.shape[0] - 1:
                    N, Naft = cal_N_Naft(tmp, [x[k], x[k+1]])
                    if Naft != 0 and N != 0:
                        y.append(np.log10(N / Naft))
                    else:
                        y.append(float('inf'))
                else:
                    N, Naft = cal_N_Naft(tmp, [x[k], float('inf')])
                    if Naft != 0 and N != 0:
                        y.append(np.log10(N / Naft))
                    else:
                        y.append(float('inf'))

            y = np.array(y)
            x, y = x[y != float('inf')], y[y != float('inf')]
            x_eny = np.zeros(x.shape[0])
            for idx in range(len(x) - 1):
                x_eny[idx] = (x[idx] + x[idx + 1]) / 2
            x_eny[-1] = x[-1] + pow(10, len(str(x[-1])))*(0.9/self.interval_num) / 2
            ax.semilogx(x_eny, y, color=color, marker=marker, mec=color, mfc='none', label=label)
        ax.axhline(1.2, ls='-.', linewidth=1, color="black")
        plot_norm(ax, xlabel, ylabel, y_lim=[-1, 4], legend_loc='upper right')

    def cal_WaitingTime(self, time_origin, time_1, time_2, xlabel, ylabel):
        fig = plt.figure(figsize=[6, 3.9], num='Distribution of waiting time')
        ax = plt.subplot()
        for [time, marker, color, label] in zip([time_1, time_2], ['p', 'h'], ['blue', 'g'],
                                                ['Population 1', 'Population 2']):
            res = []
            for i in range(time.shape[0] - 1):
                res.append(time[i+1] - time[i])
            inter, mid = self.cal_negtive_interval(res)
            xx, yy = self.cal_linear(sorted(np.array(res)), inter, mid)
            ax.loglog(xx, yy, markersize=8, marker=marker, mec=color, mfc='none', color=color, label=label)
        plot_norm(ax, xlabel, ylabel, legend_loc='upper right')

    def cal_OmoriLaw(self, tmp_origin, tmp_1, tmp_2, xlabel, ylabel):
        eny_lim = [[0.01, 0.1], [0.1, 1], [1, 10], [10, 100], [100, 1000]]
        tmp_origin, tmp_1, tmp_2 = self.cal_OmiroLaw_helper(tmp_origin, eny_lim), self.cal_OmiroLaw_helper(tmp_1, eny_lim), self.cal_OmiroLaw_helper(tmp_2, eny_lim)
        for idx, [tmp, title] in enumerate(zip([tmp_origin, tmp_1, tmp_2], ['Omori law_Whole', 'Omori law_Population 1',
                                                                            'Omori law_Population 2'])):
            fig = plt.figure(figsize=[6, 3.9], num=title)
            ax = plt.subplot()
            for i, [marker, color, label] in enumerate(zip(['>', 'o', 'p', 'h', 'H'], ['r', 'y', 'blue', 'g', 'purple'],
                                                           ['$10^{-2}aJ<E_{MS}<10^{-1}aJ$', '$10^{-1}aJ<E_{MS}<10^{0}aJ$',
                                                            '$10^{0}aJ<E_{MS}<10^{1}aJ$', '$10^{1}aJ<E_{MS}<10^{2}aJ$',
                                                            '$10^{2}aJ<E_{MS}<10^{3}aJ$'])):
                if len(tmp[i]):
                    inter, mid = self.cal_negtive_interval(tmp[i])
                    xx, yy = self.cal_linear(sorted(np.array(tmp[i])), inter, mid)
                    ax.loglog(xx, yy, markersize=8, marker=marker, mec=color, mfc='none', color=color, label=label)
            plot_norm(ax, xlabel, ylabel, legend_loc='upper right')


if __name__ == "__main__":
    path = r'E:\data\vallen'
    fold = 'Ni-tension test-pure-1-0.01-AE-20201030'
    path_pri = fold + '.pridb'
    path_tra = fold + '.tradb'
    features_path = fold + '.txt'
    os.chdir('/'.join([path, fold]))
    # 2020.11.10-PM-self
    # 6016_CR_1
    # 316L-1.5-z3-AE-3 sensor-20200530
    # Ni-tension test-electrolysis-1-0.01-AE-20201031
    # Ni-tension test-pure-1-0.01-AE-20201030
    # 2020.11.10-PM-self

    reload = Reload(path_pri, path_tra, fold)
    data_tra, data_pri, chan_1, chan_2, chan_3, chan_4 = reload.read_data(lower=2)
    print('Channel 1: {} | Channel 2: {} | Channel 3: {} | Channel 4: {}'.format(chan_1.shape[0], chan_2.shape[0],
                                                                                 chan_3.shape[0], chan_4.shape[0]))
    # SetID, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts, TRAI
    chan = chan_2
    Time, Amp, RiseT, Dur, Eny, RMS, Counts = chan[:, 1], chan[:, 4], chan[:, 5], \
                                              chan[:, 6], chan[:, 7], chan[:, 8], chan[:, 9]

    # SetID, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts, TRAI
    feature_idx = [Amp, Dur, Time]
    xlabelz = ['Amplitude (μV)', 'Duration (μs)', 'Energy (aJ)']
    ylabelz = ['PDF(A)', 'PDF(D)', 'PDF(E)']
    color_1 = [255 / 255, 0 / 255, 102 / 255]  # red
    color_2 = [0 / 255, 136 / 255, 204 / 255]  # blue
    features = Features(color_1, color_2, Time, feature_idx, 8)
    interz, midz = features.cal_interval()

    # ICA and Kernel K-Means
    S_, A_ = ICA(2, np.log10(Amp), np.log10(Eny), np.log10(Dur))
    km = KernelKMeans(n_clusters=2, max_iter=100, random_state=100, verbose=1, kernel="rbf")
    pred = km.fit_predict(S_)
    cls_KKM = []
    for i in range(2):
        cls_KKM .append(pred == i)
    cls_KKM[0], cls_KKM[1] = pred == 1, pred == 0

    # Plot log to log scatter and Selected waveform
    # features.plot_correlation(Dur, Eny, xlabelz[1], xlabelz[2], 'Chan 2', cls_KKM[0], cls_KKM[1])
    # features.plot_correlation(Dur, Amp, xlabelz[1], xlabelz[0], 'Chan 2', cls_KKM[0], cls_KKM[1])
    # features.plot_correlation(Amp, Eny, xlabelz[0], xlabelz[2], cls_KKM[0], cls_KKM[1], idx_same_amp_1, idx_same_amp_2,
    #                           title='Same amplitude')
    # features.plot_correlation(Amp, Eny, xlabelz[0], xlabelz[2], cls_KKM[0], cls_KKM[1], idx_same_eny_1, idx_same_eny_2,
    #                           title='Same energy')
    # features.plot_correlation(Amp, Eny, xlabelz[0], xlabelz[2], cls_KKM[0], cls_KKM[1])
    # features.plot_feature_time(Eny, xlabelz[2])
    # features.cal_contour(Amp, Eny, '$20 \log_{10} A(\mu V)$', '$20 \log_{10} E(aJ)$', [20, 55], [-20, 40], 50, 50)
    # features.cal_waitingTime(Eny, features_path, cls_KKM[0], cls_KKM[1], 'Δt(s)', 'p(Δt)')

    # for i, [idx, inter, mid, xlabel, ylabel] in enumerate(zip(feature_idx, interz, midz, xlabelz, ylabelz)):
    #     tmp, tmp_1, tmp_2 = sorted(idx), sorted(idx[cls_KKM[0]]), sorted(idx[cls_KKM[1]])
    #     features.cal_PDF(tmp, features_path, interz[i], midz[i], tmp_1, tmp_2, xlabel, ylabel)
    #     features.cal_CCDF(tmp, features_path, tmp_1, tmp_2, xlabel, 'CCD C(s)')
    #     features.cal_ML(tmp, features_path, tmp_1, tmp_2, xlabel, r'$\epsilon$')

    plt.show()