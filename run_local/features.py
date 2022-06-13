# -*- coding: UTF-8 -*-
from plot_format import plot_norm
from collections import Counter
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import json
from tqdm import tqdm
import array
import csv
import sqlite3
from kmeans import *
from utils import *
from wave_freq import *
from stream_old import *
import warnings
from matplotlib.pylab import mpl
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore")
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


class Features:
    def __init__(self, color_1, color_2, time, status, font='Arial', frameon=False):
        self.color_1 = color_1
        self.color_2 = color_2
        self.time = time
        self.convert = lambda x, a, b: pow(x, a) * pow(10, b)
        self.status = status
        self.frameon = frameon
        self.font = font

    def __cal_linear_interval(self, tmp, interval):
        """
        Take the linear interval to get the first number in each order and the interval between grids
        :param tmp: Energy/Amplitude/Duration in order of magnitude
        :param interval: Number of bins in each order of magnitude
        :return:
        """
        tmp_max = int(max(tmp))
        tmp_min = int(min(tmp))
        mid = []
        if tmp_min <= 0:
            inter = [0] + [pow(10, i) for i in range(len(str(tmp_max)))]
        else:
            inter = [pow(10, i) for i in range(len(str(tmp_min)) - 1, len(str(tmp_max)))]
        for idx in range(len(inter)):
            try:
                mid.extend([(inter[idx + 1] - inter[idx]) / interval])
            except IndexError:
                mid.extend([9 * inter[idx] / interval])
        return inter, mid

    def __cal_log_interval(self, tmp):
        """
        Take the logarithmic interval to get the first number in each order
        :param tmp: Energy/Amplitude/Duration in order of magnitude
        :return:
        """
        tmp_min = math.floor(np.log10(min(tmp)))
        tmp_max = math.ceil(np.log10(max(tmp)))
        inter = [i for i in range(tmp_min, tmp_max + 1)]
        return inter

    def __cal_negtive_interval(self, res, interval):
        """

        :param res:
        :param interval:
        :return:
        """
        tmp = sorted(np.array(res))
        tmp_min, tmp_max = math.floor(np.log10(min(tmp))), math.ceil(np.log10(max(tmp)))
        inter = [pow(10, i) for i in range(tmp_min, tmp_max + 1)]
        mid = [interval * pow(10, i) for i in range(tmp_min + 1, tmp_max + 2)]
        return inter, mid

    def __cal_linear(self, tmp, inter, mid, interval_num, idx=0):
        """
        Calculate the probability density value at linear interval
        :param tmp: Energy/Amplitude/Duration in order of magnitude
        :param inter: The first number of each order of magnitude
        :param mid: Bin spacing per order of magnitude
        :param interval_num: Number of bins divided in each order of magnitude
        :param idx:
        :return:
        """
        # 初始化横坐标
        x = np.array([])
        for i in inter:
            if i != 0:
                x = np.append(x, np.linspace(i, i * 10, interval_num, endpoint=False))
            else:
                x = np.append(x, np.linspace(i, 1, interval_num, endpoint=False))
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
        xx[-1] = x[-1] + pow(10, len(str(int(x[-1])))) * (0.9 / interval_num) / 2
        # 计算分段区间长度，从而求得概率密度值
        interval = []
        for i, j in enumerate(mid):
            try:
                # num = len(np.intersect1d(np.where(inter[i] <= xx)[0],
                #                          np.where(xx < inter[i + 1])[0]))
                num = len(np.where((inter[i] <= xx) & (xx < inter[i + 1]))[0])
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

    def __cal_log(self, tmp, inter, interval_num, idx=0):
        """
        Calculate the probability density value at logarithmic interval
        :param tmp: Energy/Amplitude/Duration in order of magnitude
        :param inter: The first number of each order of magnitude
        :param interval_num: Number of bins divided in each order of magnitude
        :param idx:
        :return:
        """
        x, xx, interval = np.array([]), np.array([]), np.array([])
        for i in inter:
            logspace = np.logspace(i, i + 1, interval_num, endpoint=False)
            tmp_inter = [logspace[i + 1] - logspace[i] for i in range(len(logspace) - 1)]
            tmp_xx = [(logspace[i + 1] + logspace[i]) / 2 for i in range(len(logspace) - 1)]
            tmp_inter.append(10 * logspace[0] - logspace[-1])
            tmp_xx.append((10 * logspace[0] + logspace[-1]) / 2)
            x = np.append(x, logspace)
            interval = np.append(interval, np.array(tmp_inter))
            xx = np.append(xx, np.array(tmp_xx))

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

        xx, y, interval = xx[y != 0], y[y != 0], interval[y != 0]
        yy = y / (sum(y) * interval)
        # yy = y / sum(y)
        return xx, yy

    def __cal_N_Naft(self, tmp, eny_lim):
        """
        Calculation of the number of aftershocks after the main shock.
        :param tmp: Energy
        :param eny_lim: Minimum value of each bin
        :return:
        """
        N_ms, N_as = 0, 0
        main_peak = np.where(eny_lim[0] < tmp)[0]
        if len(main_peak):
            for i in range(main_peak.shape[0] - 1):
                if main_peak[i] >= eny_lim[1]:
                    continue
                elif main_peak[i + 1] - main_peak[i] == 1:
                    N_ms += tmp[main_peak[i]]
                    continue
                N_ms += tmp[main_peak[i]]
                N_as += np.max(tmp[main_peak[i] + 1:main_peak[i + 1]])
            if main_peak[-1] < tmp.shape[0] - 1:
                N_as += np.max(tmp[main_peak[-1] + 1:])
            N_ms += tmp[main_peak[-1]]
        return N_ms + N_as, N_as

    def __cal_OmiroLaw_helper(self, tmp, eny_lim):
        """
        Calculate the probability density distribution of the corresponding interval energy
        :param tmp: Energy
        :param eny_lim: Minimum value of each bin
        :return:
        """
        res = [[] for _ in range(len(eny_lim))]
        for idx in range(len(eny_lim)):
            main_peak = np.where((eny_lim[idx][0] < tmp) & (tmp < eny_lim[idx][1]))[0]
            if len(main_peak):
                for i in range(main_peak.shape[0] - 1):
                    for j in range(main_peak[i] + 1, main_peak[i + 1] + 1):
                        if tmp[j] < eny_lim[idx][1]:
                            k = self.time[j] - self.time[main_peak[i]]
                            res[idx].append(k)
                        else:
                            break
                if main_peak[-1] < tmp.shape[0] - 1:
                    for j in range(main_peak[-1] + 1, tmp.shape[0]):
                        k = self.time[j] - self.time[main_peak[-1]]
                        res[idx].append(k)
        return res

    def __cal_OmiroLaw_timeSeq_helper(self, tmp, cls_idx):
        """
        Calculate the triggering time of the main shock to aftershocks
        :param tmp: Energy
        :param cls_idx: Label
        :return:
        """
        res = []
        main_peak = np.where(cls_idx is True)[0] if type(cls_idx[0]) == bool else cls_idx
        eny_lim = [min(tmp[cls_idx]), max(tmp[cls_idx])]
        if len(main_peak):
            for i in range(len(main_peak) - 1):
                for j in range(main_peak[i] + 1, main_peak[i + 1] + 1):
                    if tmp[j] < eny_lim[1]:
                        k = self.time[j] - self.time[main_peak[i]]
                        res.append(k)
                    else:
                        break
            if main_peak[-1] < tmp.shape[0] - 1:
                for j in range(main_peak[-1] + 1, tmp.shape[0]):
                    k = self.time[j] - self.time[main_peak[-1]]
                    res.append(k)
        return res

    def cal_PDF(self, tmp_origin, tmp_1, tmp_2, xlabel, ylabel, features_path=None, LIM=None, INTERVAL_NUM=None,
                bin_method='log', select=None, FIT=False, COLOR=None, LABEL=None):
        """
        Calculate Probability Density Distribution Function
        :param tmp_origin: Energy/Amplitude/Duration in order of magnitude of original data
        :param tmp_1: Energy/Amplitude/Duration in order of magnitude of population 1
        :param tmp_2: Energy/Amplitude/Duration in order of magnitude of population 2
        :param xlabel: 'Amplitude (μV)', 'Duration (μs)', 'Energy (aJ)'
        :param ylabel: 'PDF (A)', 'PDF (D)', 'PDF (E)'
        :param features_path: Absolute path of output data
        :param LIM: Use in function fitting, support specific values or indexes,
                    value: [0, float('inf')], [100, 900], ...
                    index: [0, None], [11, -2], ...
        :param INTERVAL_NUM: Number of bins divided in each order of magnitude
        :param bin_method: Method to divide the bin, Support linear partition and logarithmic partition
        :param select: The category displayed each time the function is called, like:
                       [0, None]: Display the original data, the first population, the second population of calculation results.
                       [1, 2]: Only display the first population of calculation results.
        :param FIT: Whether to fit parameters, support True or False
        :param COLOR: Color when drawing with original data, population I and population II respectively
        :param LABEL: Format of legend, default ['Whole', 'Pop 1', 'Pop 2']
        :return:
        """
        if INTERVAL_NUM is None:
            INTERVAL_NUM = [6] * 3
        if select is None:
            select = [0, 3]
        if LIM is None:
            LIM = [[0, None]] * 3
        if LABEL is None:
            LABEL = ['Whole', 'Pop 1', 'Pop 2']
        if COLOR is None:
            COLOR = ['black', [1, 0, 0.4], [0, 0.53, 0.8]]
        fig = plt.figure(figsize=[6, 3.9], num='PDF--%s' % xlabel)
        # fig = plt.figure(figsize=[6, 3.9])
        fig.text(0.15, 0.2, self.status, fontdict={'family': self.font, 'fontweight': 'bold', 'fontsize': 12})
        ax = plt.subplot()
        TMP = [tmp_origin, tmp_1, tmp_2]
        if LIM[0][1] == None or LIM[0][1] < 0:
            method = 'index'
        elif LIM[0][1] == float('inf') or LIM[0][1] > 0:
            method = 'value'
        for tmp, color, label, num, lim in zip(TMP[select[0]:select[1]], COLOR[select[0]:select[1]],
                                               LABEL[select[0]:select[1]],
                                               INTERVAL_NUM[select[0]:select[1]], LIM[select[0]:select[1]]):
            if bin_method == 'linear':
                inter, mid = self.__cal_linear_interval(tmp, num)
                xx, yy = self.__cal_linear(tmp, inter, mid, num)
            elif bin_method == 'log':
                inter = self.__cal_log_interval(tmp)
                xx, yy = self.__cal_log(tmp, inter, num)
            if FIT:
                if method == 'value':
                    lim = np.where((xx > lim[0]) & (xx < lim[1]))[0]
                    fit = np.polyfit(np.log10(xx[lim[0]:lim[-1]]), np.log10(yy[lim[0]:lim[-1]]), 1)
                elif method == 'index':
                    fit = np.polyfit(np.log10(xx[lim[0]:lim[1]]), np.log10(yy[lim[0]:lim[1]]), 1)
                alpha, b = fit[0], fit[1]
                fit_x = np.linspace(xx[lim[0]], xx[-1], 100)
                fit_y = self.convert(fit_x, alpha, b)
                ax.plot(fit_x, fit_y, '-.', lw=1, color=color)
                ax.loglog(xx, yy, '.', marker='.', markersize=8, color=color,
                          label='{}--{:.2f}'.format(label, abs(alpha)))
            else:
                ax.loglog(xx, yy, '.', marker='.', markersize=8, color=color, label=label)
            if features_path:
                with open(f'{features_path[:-4]}_{label}_{ylabel}.txt', 'w') as f:
                    f.write('{}, {}\n'.format(xlabel, ylabel))
                    for j in range(len(xx)):
                        f.write('{}, {}\n'.format(xx[j], yy[j]))

        plot_norm(ax, xlabel, ylabel, legend_loc='upper right', frameon=self.frameon, fontname=self.font)

    def cal_CCDF(self, tmp_origin, tmp_1, tmp_2, xlabel, ylabel, features_path=None, LIM=None, select=None, FIT=False,
                 COLOR=None, LABEL=None):
        """
        Calculate Complementary Cumulative Distribution Function
        :param tmp_origin: Energy/Amplitude/Duration in order of magnitude of original data
        :param tmp_1: Energy/Amplitude/Duration in order of magnitude of population 1
        :param tmp_2: Energy/Amplitude/Duration in order of magnitude of population 2
        :param xlabel: 'Amplitude (μV)', 'Duration (μs)', 'Energy (aJ)'
        :param ylabel: 'CCDF (A)', 'CCDF (D)', 'CCDF (E)'
        :param features_path: Absolute path of output data
        :param LIM: Use in function fitting, support specific values or indexes,
                    value: [0, float('inf')], [100, 900], ...
                    index: [0, None], [11, -2], ...
        :param select: The category displayed each time the function is called, like:
                       [0, None]: Display the original data, the first population, the second population of calculation results.
                       [1, 2]: Only display the first population of calculation results.
        :param FIT: Whether to fit parameters, support True or False
        :param COLOR: Color when drawing with original data, population I and population II respectively
        :param LABEL: Format of legend, default ['Whole', 'Pop 1', 'Pop 2']
        :return:
        """
        if LIM is None:
            LIM = [[0, float('inf')]] * 3
        if select is None:
            select = [0, 3]
        if LABEL is None:
            LABEL = ['Whole', 'Pop 1', 'Pop 2']
        if COLOR is None:
            COLOR = ['black', [1, 0, 0.4], [0, 0.53, 0.8]]
        N_origin, N1, N2 = len(tmp_origin) if tmp_origin else 0, len(tmp_1) if tmp_1 else 0, len(tmp_2) if tmp_2 else 0
        fig = plt.figure(figsize=[6, 3.9], num='CCDF--%s' % xlabel)
        fig.text(0.15, 0.2, self.status, fontdict={'family': self.font, 'fontweight': 'bold', 'fontsize': 12})
        ax = plt.subplot()
        TMP, N = [tmp_origin, tmp_1, tmp_2], [N_origin, N1, N2]
        for tmp, N, color, label, lim in zip(TMP[select[0]:select[1]], N[select[0]:select[1]],
                                             COLOR[select[0]:select[1]],
                                             LABEL[select[0]:select[1]], LIM[select[0]:select[1]]):
            xx, yy = [], []
            for i in range(N - 1):
                xx.append(np.mean([tmp[i], tmp[i + 1]]))
                yy.append((N - i + 1) / N)
            if FIT:
                xx, yy = np.array(xx), np.array(yy)
                fit_lim = np.where((xx > lim[0]) & (xx < lim[1]))[0]
                try:
                    fit = np.polyfit(np.log10(xx[fit_lim[0]:fit_lim[-1]]), np.log10(yy[fit_lim[0]:fit_lim[-1]]), 1)
                except IndexError:
                    print("Please select a correct range of 'lim_ccdf'.")
                    return
                alpha, b = fit[0], fit[1]
                fit_x = np.linspace(xx[fit_lim[0]], xx[fit_lim[-1]], 100)
                fit_y = self.convert(fit_x, alpha, b)
                ax.plot(fit_x, fit_y, '-.', lw=1, color=color)
                ax.loglog(xx, yy, color=color, label='{}--{:.2f}'.format(label, abs(alpha)))
            else:
                ax.loglog(xx, yy, color=color, label=label)
            if features_path:
                with open(f'{features_path[:-4]}_{label}_CCDF({xlabel[0]}).txt', 'w') as f:
                    f.write('{}, {}\n'.format(xlabel, ylabel))
                    for j in range(len(xx)):
                        f.write('{}, {}\n'.format(xx[j], yy[j]))
        plot_norm(ax, xlabel, ylabel, legend_loc='upper right', frameon=self.frameon, fontname=self.font)

    def cal_ML(self, tmp_origin, tmp_1, tmp_2, xlabel, ylabel, features_path=None, select=None, COLOR=None, ECOLOR=None,
               LABEL=None):
        """
        Calculate the maximum likelihood function distribution
        :param tmp_origin: Energy/Amplitude/Duration in order of magnitude of original data
        :param tmp_1: Energy/Amplitude/Duration in order of magnitude of population 1
        :param tmp_2: Energy/Amplitude/Duration in order of magnitude of population 2
        :param xlabel: 'Amplitude (μV)', 'Duration (μs)', 'Energy (aJ)'
        :param ylabel: 'ML (A)', 'ML (D)', 'ML (E)'
        :param features_path: Absolute path of output data
        :param select: The category displayed each time the function is called, like:
                       [0, None]: Display the original data, the first population, the second population of calculation results.
                       [1, 2]: Only display the first population of calculation results.
        :param COLOR: Color when drawing with original data, population I and population II respectively
        :param ECOLOR: Line color of error bar, corresponding parameter COLOR
        :param LABEL: Format of legend, default ['Whole', 'Pop 1', 'Pop 2']
        :return:
        """
        if not select:
            select = [0, 3]
        if not LABEL:
            LABEL = ['Whole', 'Pop 1', 'Pop 2']
        if not COLOR:
            COLOR = ['black', [1, 0, 0.4], [0, 0.53, 0.8]]
        if not ECOLOR:
            ECOLOR = [[0.7, 0.7, 0.7], [1, 0.58, 0.67], [0.93, 0.39, 0.93]]
        N_origin, N1, N2 = len(tmp_origin) if tmp_origin else 0, len(tmp_1) if tmp_1 else 0, len(tmp_2) if tmp_2 else 0
        # fig = plt.figure(figsize=[6, 3.9], num='ML--%s' % xlabel)
        fig = plt.figure(figsize=[6, 3.9])
        fig.text(0.96, 0.2, self.status, fontdict={'family': self.font, 'fontweight': 'bold', 'fontsize': 12},
                 horizontalalignment="right")
        ax = plt.subplot()
        ax.set_xscale("log", nonposx='clip')
        TMP, N, LAYER = [tmp_origin, tmp_1, tmp_2], [N_origin, N1, N2], [1, 3, 2]
        for tmp, N, layer, color, ecolor, label in zip(TMP[select[0]:select[1]], N[select[0]:select[1]],
                                                       LAYER[select[0]:select[1]], COLOR[select[0]:select[1]],
                                                       ECOLOR[select[0]:select[1]], LABEL[select[0]:select[1]]):
            ML_y, Error_bar = [], []
            for j in tqdm(range(N)):
                valid_x = tmp[j:]
                E0 = valid_x[0]
                Sum = np.sum(np.log(valid_x / E0)) + 1e-5
                N_prime = N - j
                alpha = 1 + N_prime / Sum
                error_bar = (alpha - 1) / pow(N_prime, 0.5)
                ML_y.append(alpha)
                Error_bar.append(error_bar)
            _, caps, bars = ax.errorbar(tmp, ML_y, yerr=Error_bar, fmt='o', color=color, ecolor=ecolor, elinewidth=1,
                                        capsize=2, ms=3, label=label, zorder=layer, alpha=0.8)
            _ = [bar.set_alpha(0.5) for bar in bars]
            _ = [cap.set_alpha(0.5) for cap in caps]
            if features_path:
                with open(f'{features_path[:-4]}_{label}_ML({xlabel[0]}).txt', 'w') as f:
                    f.write('{}, {}, Error bar\n'.format(xlabel, ylabel))
                    for j in range(len(ML_y)):
                        f.write('{}, {}, {}\n'.format(tmp[j], ML_y[j], Error_bar[j]))
        plot_norm(ax, xlabel, ylabel, y_lim=[1.3, 3.0], legend_loc='upper right', frameon=self.frameon,
                  fontname=self.font)

    def cal_contour(self, tmp_1, tmp_2, xlabel, ylabel, title, x_lim, y_lim, size_x=40, size_y=40,
                    method='linear_bin', padding=False, clabel=False):
        """
        Visualization of contour line between AE features
        :param tmp_1: Energy/Amplitude/Duration in order of magnitude of original data
        :param tmp_2: Energy/Amplitude/Duration in order of magnitude of original data
        :param xlabel: 'Amplitude (μV)', 'Duration (μs)', 'Energy (aJ)'
        :param ylabel: 'Amplitude (μV)', 'Duration (μs)', 'Energy (aJ)'
        :param title:
        :param x_lim: Range of horizontal axis
        :param y_lim: Range of vertical axis
        :param size_x: Number of bins of horizontal axis
        :param size_y: Number of bins of vertical axis
        :param method: Method to divide the bin, Support linear partition and logarithmic partition, e.g., linear_bin, log_bin
        :param padding: Whether to fill color between contour lines
        :param clabel: Whether to add height and numbers
        :return:
        """
        tmp_1, tmp_2 = 20 * np.log10(tmp_1), 20 * np.log10(tmp_2)
        if method == 'log_bin':
            sum_x, sum_y = x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]
            arry_x = np.logspace(np.log10(sum_x + 10), 1, size_x) / (
                    sum(np.logspace(np.log10(sum_x + 10), 1, size_x)) / sum_x)
            arry_y = np.logspace(np.log10(sum_y + 10), 1, size_y) / (
                    sum(np.logspace(np.log10(sum_y + 10), 1, size_y)) / sum_y)
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

        fig = plt.figure(figsize=[6, 3.9],
                         num='Contour--%s & %s' % (ylabel.split(' ')[-1][0], xlabel.split(' ')[-1][0]))
        fig.text(0.96, 0.2, self.status, fontdict={'family': self.font, 'fontweight': 'bold', 'fontsize': 12},
                 horizontalalignment="right")
        ax = plt.subplot()
        if padding:
            ct = ax.contourf(X, Y, height, levels, colors=colors, extend='max')
        #             cbar = plt.colorbar(ct)
        else:
            ct = ax.contour(X, Y, height, levels, colors=colors, linewidths=1, linestyles=linestyles)
        #             cbar = plt.colorbar(ct)
        if clabel:
            ax.clabel(ct, inline=True, colors='k', fmt='%.1f')
        plot_norm(ax, xlabel, ylabel, title=title, legend=False, frameon=self.frameon, fontname=self.font)

    def plot_correlation(self, tmp_1, tmp_2, xlabel, ylabel, cls_1=None, cls_2=None, idx_1=None, idx_2=None, fit=False,
                         status='A-D', x1_lim=None, x2_lim=None, plot_lim=None, title=''):
        """
        Visualization of Relationships Between Acoustic Emission Features
        :param tmp_1: Energy/Amplitude/Duration in order of magnitude of original data
        :param tmp_2: Energy/Amplitude/Duration in order of magnitude of original data
        :param xlabel: 'Amplitude (μV)', 'Duration (μs)', 'Energy (aJ)'
        :param ylabel: 'Amplitude (μV)', 'Duration (μs)', 'Energy (aJ)'
        :param cls_1: Label of population 1
        :param cls_2: Label of population 2
        :param idx_1: Highlight specific data in the population 1
        :param idx_2: Highlight specific data in the population 2
        :param fit: Whether to fit parameters, support True or False
        :param status: Conditions used to determine the fit, e.g., 'E-A', 'E-D', 'A-D'
        :param x1_lim: Fit range for the population 1 of data
        :param x2_lim: Fit range for the population 2 of data
        :param plot_lim: Range of fitted line
        :param title:
        :return:
        """
        fig = plt.figure(figsize=[6, 3.9], num='Correlation--%s & %s %s' % (ylabel, xlabel, title))
        fig.text(0.96, 0.2, self.status, fontdict={'family': self.font, 'fontweight': 'bold', 'fontsize': 12},
                 horizontalalignment="right")
        ax = plt.subplot()
        if cls_1 is not None and cls_2 is not None:
            ax.loglog(tmp_1[cls_1], tmp_2[cls_1], '.', marker='.', markersize=8, color=self.color_1,
                      label='Pop 1')
            ax.loglog(tmp_1[cls_2], tmp_2[cls_2], '.', marker='.', markersize=8, color=self.color_2,
                      label='Pop 2')
            if idx_1 is not None:
                ax.loglog(tmp_1[cls_1][idx_1], tmp_2[cls_1][idx_1], '.', marker='.', markersize=8, color='black')
            if idx_2 is not None:
                ax.loglog(tmp_1[cls_2][idx_2], tmp_2[cls_2][idx_2], '.', marker='.', markersize=8, color='black')
            plot_norm(ax, xlabel, ylabel, frameon=self.frameon, fontname=self.font)
        else:
            ax.loglog(tmp_1, tmp_2, '.', Marker='.', markersize=8, color='b')
            plot_norm(ax, xlabel, ylabel, legend=False, frameon=self.frameon, fontname=self.font)

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
            ave = 0
            alpha, b, fit_x, fit_y = [], [], [], []
            mix_cor_x = [min(cor_x1), min(cor_x2)] if status == 'E-A' else plot_lim
            for linear_x, linear_y, min_x, max_x in zip([linear_x1, linear_x2], [linear_y1, linear_y2],
                                                        mix_cor_x, [max(cor_x1), max(cor_x2)]):
                fit = np.polyfit(np.log10(linear_x), np.log10(linear_y), 1)
                alpha.append(fit[0])
                b.append(fit[1])
                fit_x.append(np.linspace(min_x, max_x, 100))
                fit_y.append(self.convert(np.linspace(min_x, max_x, 100), fit[0], fit[1]))
            ax.plot(fit_x[0], fit_y[0], ls='--', lw=2, color='black')
            ax.plot(fit_x[1], fit_y[1], ls='--', lw=2, color='black')
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

    def plot_multi_correlation(self, tmp_1, tmp_2, cls_idx, xlabel, ylabel, fig_loc=None, color=None, sharex=True,
                               sharey=True):
        """

        :param tmp_1:
        :param tmp_2:
        :param cls_idx:
        :param xlabel:
        :param ylabel:
        :param fig_loc: Group image format, e.g., [3, 1]
        :param color: Color when drawing with original data, population I and population II respectively
        :param sharex: Whether to share the horizontal axis
        :param sharey: Whether to share the vertical axis
        :return:
        """
        if fig_loc is None:
            fig_loc = [3, 1]
        if not color:
            color = [self.color_1, self.color_2, 'purple']
        else:
            assert len(fig_loc) == len(color), \
                print("Length of parameter 'fig_loc' should be equal to length of parameter 'color'.")
        fig, axes = plt.subplots(fig_loc[0], fig_loc[1], sharex=sharex, sharey=sharey, figsize=(6, 9))
        if axes.ndim == 1:
            for idx, ax in enumerate(axes):
                ax.semilogy(tmp_1[cls_idx[idx]], tmp_2[cls_idx[idx]], '.', Marker='.', color=color[idx],
                            label='Pop %d' % (idx + 1))
                plot_norm(ax, xlabel, ylabel, legend=True, frameon=self.frameon, fontname=self.font)
        else:
            for idx, axs in enumerate(axes):
                for idy, ax in enumerate(axs):
                    ax.semilogy(tmp_1[cls_idx[idx]], tmp_2[cls_idx[idx]], '.', Marker='.', color=color[idx],
                                label='Pop %d' % (idx * fig_loc[1] + idy + 1))
                    plot_norm(ax, xlabel, ylabel, legend=True, frameon=self.frameon, fontname=self.font)
        plt.subplots_adjust(wspace=0, hspace=0)

    def plot_3D_correlation(self, tmp_1, tmp_2, tmp_3, xlabel, ylabel, zlabel, cls_1=None, cls_2=None, idx_1=None,
                            idx_2=None, title=''):
        """
        3D visualization of amplitude-energy-duration
        :param tmp_1: Amplitude of original data
        :param tmp_2: Energy of population 1
        :param tmp_3: Duration of population 2
        :param xlabel: 'Amplitude (μV)'
        :param ylabel: 'Energy (aJ)'
        :param zlabel: 'Duration (μs)'
        :param cls_1: Label of population 1
        :param cls_2: Label of population 2
        :param idx_1: Highlight specific data in the population 1
        :param idx_2: Highlight specific data in the population 2
        :param title:
        :return:
        """
        fig = plt.figure(figsize=[6, 3.9], num='3D Correlation--%s & %s %s' % (xlabel, ylabel, zlabel))
        ax = plt.subplot(projection='3d')
        if cls_1 is not None and cls_2 is not None:
            ax.scatter3D(np.log10(tmp_1)[cls_1], np.log10(tmp_2)[cls_1], np.log10(tmp_3)[cls_1], s=15,
                         color=self.color_1)
            ax.scatter3D(np.log10(tmp_1)[cls_2], np.log10(tmp_2)[cls_2], np.log10(tmp_3)[cls_2], s=15,
                         color=self.color_2)
            if idx_1 is not None:
                ax.scatter3D(np.log10(tmp_1)[cls_1][idx_1], np.log10(tmp_2)[cls_1][idx_1],
                             np.log10(tmp_3)[cls_1][idx_1], s=15, color='black')
            if idx_2 is not None:
                ax.scatter3D(np.log10(tmp_1)[cls_2][idx_2], np.log10(tmp_2)[cls_2][idx_2],
                             np.log10(tmp_3)[cls_2][idx_2], s=15, color='black')
        else:
            ax.scatter3D(np.log10(tmp_1), np.log10(tmp_2), np.log10(tmp_3), s=15, color=self.color_1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter('$10^{:.0f}$'.format))
        ax.yaxis.set_major_formatter(plt.FuncFormatter('$10^{:.0f}$'.format))
        ax.zaxis.set_major_formatter(plt.FuncFormatter('$10^{:.0f}$'.format))
        plot_norm(ax, xlabel, ylabel, zlabel, title, legend=False, frameon=self.frameon, fontname=self.font)

    def plot_feature_time(self, tmp_1, tmp_2, ylabel, cls_1=None, cls_2=None, idx_1=None, idx_2=None, mode='scatter',
                          width=55):
        """
        Feature - time figure visualization
        :param tmp_1: Time
        :param tmp_2: Energy/Amplitude/Duration in order of magnitude of original data
        :param ylabel: 'Amplitude (μV)', 'Duration (μs)', 'Energy (aJ)'
        :param cls_1: Label of Energy/Amplitude/Duration of population 1
        :param cls_2: Label of Energy/Amplitude/Duration of population 2
        :param idx_1: Highlight specific data in the population 1
        :param idx_2: Highlight specific data in the population 2
        :param mode: Visualize as a histogram or scatterplot, e.g., 'scatter', 'bar'
        :param width: the width of each column, For histograms only and in combination with [:param mode].
        :return:
        """
        fig = plt.figure(figsize=[6, 3.9], num='Time domain curve')
        fig.text(0.96, 0.2, self.status, fontdict={'family': self.font, 'fontweight': 'bold', 'fontsize': 12},
                 horizontalalignment="right")
        ax = plt.subplot()
        if mode == 'bar':
            if cls_1 is not None and cls_2 is not None:
                ax.bar(tmp_1[cls_1], tmp_2[cls_1], color=self.color_1, width=width, log=True)
                ax.bar(tmp_1[cls_2], tmp_2[cls_2], color=self.color_2, width=width, log=True)
                if idx_1 is not None:
                    ax.bar(tmp_1[cls_1][idx_1], tmp_2[cls_1][idx_1], color='black', width=width, log=True)
                if idx_2 is not None:
                    ax.bar(tmp_1[cls_2][idx_2], tmp_2[cls_2][idx_2], color='black', width=width, log=True)
            else:
                ax.bar(tmp_1, tmp_2, color='b', width=width, log=True)
        elif mode == 'scatter':
            if cls_1 is not None and cls_2 is not None:
                ax.semilogy(tmp_1[cls_1], tmp_2[cls_1], '.', Marker='.', color=self.color_1)
                ax.semilogy(tmp_1[cls_2], tmp_2[cls_2], '.', Marker='.', color=self.color_2)
                if idx_1 is not None:
                    ax.semilogy(tmp_1[cls_1][idx_1], tmp_2[cls_1][idx_1], '.', Marker='.', color='black')
                if idx_2 is not None:
                    ax.semilogy(tmp_1[cls_2][idx_2], tmp_2[cls_2][idx_2], '.', Marker='.', color='black')
            else:
                ax.semilogy(tmp_1, tmp_2, '.', Marker='.', color='b')
        plot_norm(ax, 'Time (s)', ylabel, legend=False, frameon=self.frameon, fontname=self.font)

    def cal_BathLaw(self, tmp_origin, tmp_1, tmp_2, xlabel, ylabel, INTERVAL_NUM=None, bin_method='log', select=None,
                    COLOR=None, LABEL=None):
        """
        Calculate the bath law for different classes of energy
        :param tmp_origin: Energy in order of magnitude of original data
        :param tmp_1: Energy in order of magnitude of population 1
        :param tmp_2: Energy in order of magnitude of population 2
        :param xlabel: 'Mainshock Energy (aJ)'
        :param ylabel: r'$\mathbf{\Delta}$M'
        :param INTERVAL_NUM: Number of bins divided in each order of magnitude
        :param bin_method: Method to divide the bin, Support linear partition and logarithmic partition
        :param select:The category displayed each time the function is called, like:
                       [0, None]: Display the original data, the first population, the second population of calculation results.
                       [1, 2]: Only display the first population of calculation results.
        :param COLOR: Color when drawing with original data, population I and population II respectively
        :param LABEL: Format of legend, default ['Whole', 'Pop 1', 'Pop 2']
        :return:
        """
        if select is None:
            select = [0, 3]
        if INTERVAL_NUM is None:
            INTERVAL_NUM = [8] * 3
        if LABEL is None:
            LABEL = ['Whole', 'Pop 1', 'Pop 2']
        if COLOR is None:
            COLOR = ['black', [1, 0, 0.4], [0, 0.53, 0.8]]
        fig = plt.figure(figsize=[6, 3.9], num='Bath law')
        #         fig = plt.figure(figsize=[6, 3.9])
        fig.text(0.12, 0.2, self.status, fontdict={'family': self.font, 'fontweight': 'bold', 'fontsize': 12})
        ax = plt.subplot()
        TMP, MARKER = [tmp_origin, tmp_1, tmp_2], ['o', 'p', 'h']

        for tmp, interval_num, marker, color, label in zip(TMP[select[0]:select[1]], INTERVAL_NUM[select[0]:select[1]],
                                                           MARKER[select[0]:select[1]], COLOR[select[0]:select[1]],
                                                           LABEL[select[0]:select[1]]):
            tmp_max = int(max(tmp))
            if bin_method == 'linear':
                x = np.array([])
                inter = [pow(10, i) for i in range(0, len(str(tmp_max)))]
                for i in inter:
                    x = np.append(x, np.linspace(i, i * 10, interval_num, endpoint=False))
            elif bin_method == 'log':
                x, x_eny = np.array([]), np.array([])
                inter = self.__cal_log_interval(tmp)
                for i in inter:
                    if i < 0:
                        continue
                    logspace = np.logspace(i, i + 1, interval_num, endpoint=False)
                    x = np.append(x, logspace)
                    tmp_xx = [(logspace[i + 1] + logspace[i]) / 2 for i in range(len(logspace) - 1)]
                    tmp_xx.append((10 * logspace[0] + logspace[-1]) / 2)
                    x_eny = np.append(x_eny, np.array(tmp_xx))
            y = []
            for k in range(x.shape[0]):
                N, Naft = self.__cal_N_Naft(tmp, [x[k], x[k + 1]]) if k != x.shape[0] - 1 else \
                    self.__cal_N_Naft(tmp, [x[k], float('inf')])
                if Naft != 0 and N != 0:
                    y.append(np.log10(N / Naft))
                else:
                    y.append(float('inf'))
            y = np.array(y)
            if bin_method == 'linear':
                x, y = x[y != float('inf')], y[y != float('inf')]
                x_eny = np.zeros(x.shape[0])
                for idx in range(len(x) - 1):
                    x_eny[idx] = (x[idx] + x[idx + 1]) / 2
                x_eny[-1] = x[-1] + pow(10, len(str(int(x[-1])))) * (0.9 / interval_num) / 2
            elif bin_method == 'log':
                x_eny, y = x_eny[y != float('inf')], y[y != float('inf')]
            ax.semilogx(x_eny, y, color=color, marker=marker, markersize=8, mec=color, mfc='none', label=label)
        ax.axhline(1.2, ls='-.', linewidth=1, color="black")
        plot_norm(ax, xlabel, ylabel, y_lim=[-1, 4], legend_loc='upper right', frameon=self.frameon, fontname=self.font)

    def cal_WaitingTime(self, time_origin, time_1, time_2, dur_origin, dur_1, dur_2, xlabel, ylabel, INTERVAL_NUM=None,
                        bin_method='log', select=None, FIT=False, LIM=None, COLOR=None, LABEL=None, features_path=None):
        """
        Calculate the waiting time distribution for different classes of times
        :param time_origin: Time in order of magnitude of original data
        :param time_1: Time in order of magnitude of population 1
        :param time_2: Time in order of magnitude of population 2
        :param dur_origin: Duration in order of magnitude of original data
        :param dur_1: Duration in order of magnitude of population 1
        :param dur_2: Duration in order of magnitude of population 2
        :param xlabel: r'$\mathbf{\Delta}$t (s)'
        :param ylabel: r'P($\mathbf{\Delta}$t)'
        :param INTERVAL_NUM: Number of bins divided in each order of magnitude
        :param bin_method: Method to divide the bin, Support linear partition and logarithmic partition
        :param select: The category displayed each time the function is called, like:
                       [0, None]: Display the original data, the first population, the second population of calculation results.
                       [1, 2]: Only display the first population of calculation results.
        :param FIT: Whether to fit parameters, support True or False
        :param LIM: Use in function fitting, support specific values or indexes,
                    value: [0, float('inf')], [100, 900], ...
                    index: [0, None], [11, -2], ...
        :param COLOR: Color when drawing with original data, population I and population II respectively
        :param LABEL: Format of legend, default ['Whole', 'Pop 1', 'Pop 2']
        :param features_path: Absolute path of output data
        :return:
        """
        if INTERVAL_NUM is None:
            INTERVAL_NUM = [8] * 3
        if select is None:
            select = [0, 3]
        if LIM is None:
            LIM = [[0, None]] * 3
        if LABEL is None:
            LABEL = ['Whole', 'Pop 1', 'Pop 2']
        if COLOR is None:
            COLOR = ['black', [1, 0, 0.4], [0, 0.53, 0.8]]
        fig = plt.figure(figsize=[6, 3.9], num='Distribution of waiting time')
        # fig = plt.figure(figsize=[6, 3.9])
        fig.text(0.16, 0.22, self.status, fontdict={'family': self.font, 'fontweight': 'bold', 'fontsize': 12})
        ax = plt.subplot()
        TIME, DUR, MARKER = [time_origin, time_1, time_2], [dur_origin, dur_1, dur_2], ['o', 'p', 'h']

        for [time, dur, interval_num, marker, color, label, lim] in zip(TIME[select[0]:select[1]],
                                                                        DUR[select[0]:select[1]],
                                                                        INTERVAL_NUM[select[0]:select[1]],
                                                                        MARKER[select[0]:select[1]],
                                                                        COLOR[select[0]:select[1]],
                                                                        LABEL[select[0]:select[1]],
                                                                        LIM[select[0]:select[1]]):
            if lim[1] == None or lim[1] < 0:
                method = 'index'
            elif lim[1] == float('inf') or lim[1] > 0:
                method = 'value'
            res = list(time[1:] - (time[:-1] + dur[:-1] / 1e6))
            valid = np.where(np.array(res) > 0)[0]
            res = np.array(res)[valid].tolist()
            if bin_method == 'linear':
                inter, mid = self.__cal_negtive_interval(res, 0.9 / interval_num)
                xx, yy = self.__cal_linear(sorted(np.array(res)), inter, mid, interval_num)
            elif bin_method == 'log':
                inter = self.__cal_log_interval(res)
                xx, yy = self.__cal_log(sorted(np.array(res)), inter, interval_num)
            if FIT:
                xx, yy = np.array(xx), np.array(yy)
                if method == 'value':
                    lim = np.where((xx > lim[0]) & (xx < lim[1]))[0]
                    fit = np.polyfit(np.log10(xx[lim[0]:lim[-1]]), np.log10(yy[lim[0]:lim[-1]]), 1)
                elif method == 'index':
                    fit = np.polyfit(np.log10(xx[lim[0]:lim[1]]), np.log10(yy[lim[0]:lim[1]]), 1)
                alpha, b = fit[0], fit[1]
                fit_x = np.linspace(xx[lim[0]], xx[-1], 100)
                fit_y = self.convert(fit_x, alpha, b)
                ax.plot(fit_x, fit_y, '-.', lw=1, color=color)
                ax.loglog(xx, yy, '.', markersize=8, marker=marker, mec=color, mfc='none', color=color,
                          label='{}--{:.2f}'.format(label, abs(alpha)))
            else:
                ax.loglog(xx, yy, '.', markersize=8, marker=marker, mec=color, mfc='none', color=color, label=label)

            if features_path:
                with open(f'{features_path[:-4]}_{label}_WaitingTime.txt', 'w') as f:
                    f.write('{}, {}\n'.format(xlabel, ylabel))
                    for j in range(xx.shape[0]):
                        f.write('{}, {}\n'.format(xx[j], yy[j]))

        plot_norm(ax, xlabel, ylabel, legend_loc='upper right', frameon=self.frameon, fontname=self.font)

    def cal_OmoriLaw(self, tmp_origin, tmp_1, tmp_2, xlabel, ylabel, INTERVAL_NUM=None, bin_method='log', select=None,
                     FIT=False, features_path=None):
        """
        Calculate the probability density distribution of different classes and different interval energies
        :param tmp_origin: Energy in order of magnitude of original data
        :param tmp_1: Energy in order of magnitude of population 1
        :param tmp_2: Energy in order of magnitude of population 2
        :param xlabel: 'Energy (aJ)'
        :param ylabel: 'PDF (E)'
        :param INTERVAL_NUM: Number of bins divided in each order of magnitude
        :param bin_method: Method to divide the bin, Support linear partition and logarithmic partition
        :param select: The category displayed each time the function is called, like:
                       [0, None]: Display the original data, the first population, the second population of calculation results.
                       [1, 2]: Only display the first population of calculation results.
        :param FIT: Whether to fit parameters, support True or False
        :param features_path: Absolute path of output data
        :return:
        """
        if select is None:
            select = [0, 3]
        if INTERVAL_NUM is None:
            INTERVAL_NUM = [8] * 3
        # eny_lim = [[0.01, 0.1], [0.1, 1], [1, 10], [10, 1000], [1000, 10000]]
        eny_lim = [[0.1, 1], [1, 10], [10, 100], [100, 1000]]
        Marker = ['>', 'o', 'p', 'h', 'H'][:len(eny_lim)]
        Color = [[1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0.5, 0.5, 0.5]][:len(eny_lim)]
        Label = ['$10^{-1}aJ<E_{MS}<10^{0}aJ$', '$10^{0}aJ<E_{MS}<10^{1}aJ$', '$10^{1}aJ<E_{MS}<10^{2}aJ$',
                 '$10^{2}aJ<E_{MS}<10^{3}aJ$', '$10^{3}aJ<E_{MS}<10^{4}aJ$'][:len(eny_lim)]
        tmp_origin, tmp_1, tmp_2 = self.__cal_OmiroLaw_helper(tmp_origin, eny_lim), \
                                   self.__cal_OmiroLaw_helper(tmp_1, eny_lim), self.__cal_OmiroLaw_helper(tmp_2,
                                                                                                          eny_lim)
        TMP, TITLE = [tmp_origin, tmp_1, tmp_2], ['Omori law_Whole', 'Omori law_Pop 1', 'Omori law_Pop 2']
        for idx, [tmp, interval_num, title] in enumerate(
                zip(TMP[select[0]:select[1]], INTERVAL_NUM[select[0]:select[1]], TITLE[select[0]:select[1]])):
            fig = plt.figure(figsize=[6, 3.9], num=title)
            fig.text(0.16, 0.21, self.status, fontdict={'family': self.font, 'fontweight': 'bold', 'fontsize': 12})
            ax = plt.subplot()
            for i, [marker, color, label] in enumerate(zip(Marker, Color, Label)):
                valid = np.where(np.array(tmp[i]) > 0)[0]
                tmp[i] = np.array(tmp[i])[valid].tolist()
                if len(tmp[i]):
                    if bin_method == 'linear':
                        inter, mid = self.__cal_negtive_interval(tmp[i], 0.9 / interval_num)
                        xx, yy = self.__cal_linear(sorted(np.array(tmp[i])), inter, mid, interval_num)
                    elif bin_method == 'log':
                        inter = self.__cal_log_interval(tmp[i])
                        xx, yy = self.__cal_log(sorted(np.array(tmp[i])), inter, interval_num)
                    if FIT:
                        xx, yy = np.array(xx), np.array(yy)
                        #                         fit_lim = np.where((xx > lim[0]) & (xx < lim[1]))[0]
                        fit = np.polyfit(np.log10(xx), np.log10(yy), 1)
                        alpha, b = fit[0], fit[1]
                        fit_x = np.linspace(xx[0], xx[-1], 100)
                        fit_y = self.convert(fit_x, alpha, b)
                        ax.plot(fit_x, fit_y, '-.', lw=1, color=color)
                        ax.loglog(xx, yy, markersize=8, marker=marker, mec=color, mfc='none', color=color,
                                  label='{}--{:.2f}'.format(label, abs(alpha)))
                    else:
                        ax.loglog(xx, yy, markersize=8, marker=marker, mec=color, mfc='none', color=color, label=label)

                    if features_path:
                        with open(f"{features_path[:-4]}_{title}_{label.replace('<', ' ').replace('>', ' ')}.txt", 'w') as f:
                            f.write('t-t_{MS} (s), r_{AS}(t-t_{MS})(s^{-1})\n')
                            for j in range(xx.shape[0]):
                                f.write('{}, {}\n'.format(xx[j], yy[j]))

            plot_norm(ax, xlabel, ylabel, legend_loc='upper right', frameon=self.frameon, fontname=self.font)

    def cal_OmoriLaw_timeSeq(self, tmp_origin, cls_idx_1, cls_idx_2, INTERVAL_NUM=None, bin_method='log', FIT=False,
                             features_path=None):
        """
        Calculate the trigger relationship of two types of energy
        :param tmp_origin: Energy
        :param cls_idx_1: Label of class 1
        :param cls_idx_2: Label of class 2
        :param INTERVAL_NUM: Number of bins divided in each order of magnitude
        :param bin_method: Method to divide the bin, Support linear partition and logarithmic partition
        :param FIT: Whether to fit parameters, support True or False
        :param features_path: Absolute path of output data
        :return:
        """
        if INTERVAL_NUM is None:
            INTERVAL_NUM = [3] * 2
        res_1 = self.__cal_OmiroLaw_timeSeq_helper(tmp_origin, cls_idx_1)
        res_2 = self.__cal_OmiroLaw_timeSeq_helper(tmp_origin, cls_idx_2)
        for res, interval_num, ylabel, title in zip([res_1, res_2], INTERVAL_NUM,
                                                    [r'$\mathbf{n_1^{2}\;(t)}$', r'$\mathbf{n_2^{1}\;(t)}$'],
                                                    ['Time sequence_Population 1 as Mainshock',
                                                     'Time sequence_Population 2 as Mainshock']):
            fig = plt.figure(figsize=[6, 3.9], num=title)
            fig.text(0.16, 0.21, self.status, fontdict={'family': self.font, 'fontweight': 'bold', 'fontsize': 12})
            ax = plt.subplot()
            valid = np.where(np.array(res) > 0)[0]
            res = np.array(res)[valid].tolist()
            if len(res):
                if bin_method == 'linear':
                    inter, mid = self.__cal_negtive_interval(res, 0.9 / interval_num)
                    xx, yy = self.__cal_linear(sorted(np.array(res)), inter, mid, interval_num)
                elif bin_method == 'log':
                    inter = self.__cal_log_interval(res)
                    xx, yy = self.__cal_log(sorted(np.array(res)), inter, interval_num)
                if FIT:
                    xx, yy = np.array(xx), np.array(yy)
                    fit = np.polyfit(np.log10(xx), np.log10(yy), 1)
                    alpha, b = fit[0], fit[1]
                    fit_x = np.linspace(xx[0], xx[-1], 100)
                    fit_y = self.convert(fit_x, alpha, b)
                    ax.plot(fit_x, fit_y, '-.', lw=1, color='g', label='Slope = {:.2f}'.format(abs(alpha)))
                    ax.loglog(xx, yy, '.', markersize=8, marker='o', mec='g', mfc='none', color='g')
                else:
                    ax.loglog(xx, yy, '.', markersize=8, marker='o', mec='g', mfc='none', color='g')

            if features_path:
                with open(f'{features_path[:-4]}_{title}.txt', 'w') as f:
                    f.write(f'Time (s), {ylabel}\n')
                    for j in range(xx.shape[0]):
                        f.write('{}, {}\n'.format(xx[j], yy[j]))

            plot_norm(ax, 'Time (s)', ylabel, legend_loc='upper right', frameon=self.frameon, fontname=self.font)


if __name__ == "__main__":
    """
    AE数据的说明文件需放在此脚本同目录下，命名为“metarialsInfo.json”
    格式：
        {
          "316L": {
            "AM-Cu-20220328-test1-tension-0.05mm-min": {
              "t_str": 0,
              "t_cut": "inf"
            }
        }
    说明：
        1. 先按不同材料类别进行划分，key为材料类别，value字典用于存储对应材料的AE数据信息
        2. 同一材料中将字典的key命名为数据库名字，value字典包含但不限于"t_str"和"t_cut"两个参数。
            "t_str"参数只能为整数，用于特定时间范围筛选的起始时刻。
            "t_cut"参数只能为整数和"inf"，用于特定时间范围筛选的终止时刻。
    """
    with open('./metarialsInfo.json', 'r', encoding='utf-8') as f:
        js = json.load(f)

    path = r'F:\VALLEN\ZPH'
    fold = "tini50.8-cw40-aged400-1h-1-2-AE-0.05"
    info = js['Fe'][fold]
    path_pri = fold + '.pridb'
    path_tra = fold + '.tradb'
    features_path = fold + '.txt'
    os.chdir('/'.join([path, fold]))

    # ================================================= 说明文件入参检测 ==================================================
    try:
        for param in ['t_str', 't_cut']:
            if param in info.keys():
                if (type(info[param]) not in [int, float]) and (info[param] != 'inf'):
                    raise Exception(
                        f"Check the type of the '{param}' input parameter in the database specification file.")
            else:
                raise Exception(f"No '{param}' parameter in the database specification file.")
    except Exception as e:
        print(e)
        sys.exit(0)

    reload = Reload(path_pri, path_tra, fold)
    data_tra, data_pri, chan_1, chan_2, chan_3, chan_4 = reload.read_vallen_data(lower=2, mode='all',
                                                                                 t_str=info['t_str'],
                                                                                 t_cut=info['t_cut'] if type(
                                                                                     info['t_cut']) == int else float(
                                                                                     info['t_cut']))
    # data_tra, data_pri, chan_1, chan_2, chan_3, chan_4 = reload.read_stream_data(mode='all', t_cut=float('inf'))
    print('Channel 1: {} | Channel 2: {} | Channel 3: {} | Channel 4: {}'.format(chan_1.shape[0], chan_2.shape[0],
                                                                                 chan_3.shape[0], chan_4.shape[0]))
    # '''
    # # SetID, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts, TRAI
    # chan = chan_2
    # Time, Amp, RiseT, Dur, Eny, RMS, Counts, TRAI = chan[:, 1], chan[:, 4], chan[:, 5], chan[:, 6], chan[:, 7], \
    #                                                 chan[:, 8], chan[:, 9], chan[:, -1].astype(int)

    # # SetID, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts, TRAI
    # feature_idx = [Amp, Dur, Eny]
    # xlabelz = ['Amplitude (μV)', 'Duration (μs)', 'Energy (aJ)']
    # color_1 = [255 / 255, 0 / 255, 102 / 255]  # red
    # color_2 = [0 / 255, 136 / 255, 204 / 255]  # blue
    # status = fold.split('-')[0] + '-' + fold.split('-')[2]
    # features = Features(color_1, color_2, Time, status)

    # # ICA and Kernel K-Means
    # S_, A_ = ICA(2, np.log10(Amp), np.log10(Eny), np.log10(Dur))
    # km = KernelKMeans(n_clusters=2, max_iter=100, random_state=100, verbose=1, kernel="rbf")
    # pred = km.fit_predict(S_)
    # cls_KKM = []
    # for i in range(2):
    #     cls_KKM.append(pred == i)
    # cls_KKM[0], cls_KKM[1] = pred == 1, pred == 0

    # waveform = Waveform(color_1, color_2, data_tra, path, path_pri, status, 'vallen')
    # frequency = Frequency(color_1, color_2, data_tra, path, path_pri, status, 'vallen')

    # PackEny = []
    # for trai in tqdm(chan[:, -1].astype(int)):
    #     _, sig = waveform.cal_wave(data_tra[trai - 1])
    #     _, _, energy = frequency.cla_wtpacket(sig, 'db8', 3, False)
    #     PackEny.append([i / sum(energy) for i in energy])
    # PackEny = np.array(PackEny)

    # freq, stage_idx = frequency.cal_freq_max(chan[:, -1].astype(int), 0, status='peak')

    # df = pd.DataFrame(
    #     {'Amp': Amp, 'RiseT': RiseT, 'Dur': Dur, 'Eny': Eny, 'RMS': RMS, 'Counts': Counts, 'PeakFreq': freq,
    #      'PackEny1': PackEny[:, 0], 'PackEny2': PackEny[:, 1], 'PackEny3': PackEny[:, 2], 'PackEny4': PackEny[:, 3],
    #      'PackEny5': PackEny[:, 4], 'PackEny6': PackEny[:, 5], 'PackEny7': PackEny[:, 6], 'PackEny8': PackEny[:, 7],
    #      'Pop': cls_KKM[0].astype(int)})
    # df.to_csv('Ni_electrolysis_chan2.csv', index=None)

    # for trai, title in zip([TRAI_all, TRAI_1_all, TRAI_2_all], ['Whole', 'Population 1', 'Population 2']):
    #     Res, N = frequency.cal_ave_freq(trai, valid=False, t_lim=50)
    #     frequency.plot_ave_freq(Res, N, title)

    # for idx, lim_pdf, lim_ccdf, inerval_num in zip([0, 1, 2], LIM_PDF, LIM_CCDF, INTERVAL_NUM):
    #     tmp, tmp_1, tmp_2 = sorted(feature_idx[idx]), sorted(feature_idx[idx][cls_KKM[0]]), sorted(feature_idx[idx][cls_KKM[1]])
    #     features.cal_PDF(tmp, tmp_1, tmp_2, xlabelz[idx], 'PDF (%s)' % xlabelz[idx][0], features_path, lim_pdf, inerval_num, bin_method='log', select=[1, None], FIT=True)
    #     features.cal_ML(tmp, tmp_1, tmp_2, xlabelz[idx], 'ML (%s)' % xlabelz[idx][0], features_path, select=[1, None])
    #     features.cal_CCDF(tmp, tmp_1, tmp_2, xlabelz[idx], 'CCD C(s)', features_path, lim_ccdf, select=[1, None], FIT=True)

    # features.cal_contour(Amp, Eny, '$20 \log_{10} A(\mu V)$', '$20 \log_{10} E(aJ)$', 'Contour', [20, 55], [-20, 40], 50, 50, method='log_bin')
    # features.cal_BathLaw(Eny, Eny[cls_KKM[0]], Eny[cls_KKM[1]], 'Mainshock Energy (aJ)', r'$\mathbf{\Delta}$M', [8, 15, 15], bin_method='log', select=[1, None])
    # features.cal_WaitingTime(Time, Time[cls_KKM[0]], Time[cls_KKM[1]], Dur, Dur[cls_KKM[0]], Dur[cls_KKM[1]], r'$\mathbf{\Delta}$t (s)', r'P($\mathbf{\Delta}$t)', [8, 22, 26], bin_method='log', select=[1, None], FIT=True)
    # features.cal_OmoriLaw(Eny, Eny[cls_KKM[0]], Eny[cls_KKM[1]], r'$\mathbf{t-t_{MS}\;(s)}$', r'$\mathbf{r_{AS}(t-t_{MS})\;(s^{-1})}$', [8, 36, 19], bin_method='log', select=[1, None], FIT=True)
    # features.cal_OmoriLaw_timeSeq(Eny, cls_KKM[0], cls_KKM[1], INTERVAL_NUM=[2, 4], bin_method='log', FIT=True)
    # waveform.plot_envelope([TRAI[idx_1], TRAI[idx_crack], TRAI[idx_rotation]], ['black', color_1, color_2], features_path)
    # ave, alpha, b, A, B = features.plot_correlation(Dur, Amp, xlabelz[0], xlabelz[2], cls_1=cls_KKM[0], cls_2=cls_KKM[1], status='A-D', x1_lim=[pow(10, 2.75), float('inf')],
    #                                                 x2_lim=[pow(10, 1.7), pow(10, 2.0)], plot_lim=[150, 30], fit=True)
    # features.plot_correlation(Dur, Amp, xlabelz[1], xlabelz[0], cls_KKM[0], cls_KKM[1])
    # features.plot_correlation(Dur, Eny, xlabelz[1], xlabelz[2], cls_KKM[0], cls_KKM[1])
    # features.plot_correlation(Amp, Eny, xlabelz[0], xlabelz[2], cls_KKM[0], cls_KKM[1])

    # ------------------------------------------------------------------------------------------------------------------
    # from sklearn.svm import SVC
    # from sklearn.model_selection import train_test_split
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.ensemble import RandomForestClassifier
    # import pandas as pd
    # import seaborn as sns
    # from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_curve, auc, confusion_matrix
    # fold = r'C:\Users\Yuan\Desktop\Ni dataset\Ni\Ni_pure.csv'
    # data = pd.read_csv(fold).astype(np.float32)
    # feature = data.iloc[:, :-1].values
    # label = np.array(data.iloc[:, -1].tolist()).reshape(-1, 1)
    # # ext = np.zeros([label.shape[0], 1]).astype(np.float32)
    # # ext[np.where(label == 0)[0]] = 1
    # # label = np.concatenate((label, ext), axis=1)
    #
    # df_temp = train_test_split(feature, label, test_size=0.2, stratify=label, random_state=69)
    # stdScaler = StandardScaler().fit(df_temp[0])
    # trainStd = stdScaler.transform(df_temp[0])
    # testStd = stdScaler.transform(df_temp[1])
    #
    # svm = SVC(max_iter=200, random_state=100).fit(trainStd, df_temp[2].reshape(-1))
    # print('建立的SVM模型为：\n', svm)
    #
    # rf = RandomForestClassifier(max_depth=10, random_state=100).fit(trainStd, df_temp[2].reshape(-1))
    # print('建立的RF模型为：\n', rf)
    # fold = r'C:\Users\Yuan\Desktop\Ni dataset\Ni\Ni_electrolysis_chan2_pop2.csv'
    # data = pd.read_csv(fold).astype(np.float32)
    # nano_ni = data.values
    # stdScaler = StandardScaler().fit(nano_ni)
    # trainStd = stdScaler.transform(nano_ni)
    # target_pred = svm.predict(trainStd)

    # fig = plt.figure(figsize=[6, 3.9])
    # fig.text(0.96, 0.2, status, fontdict={'family': 'Arial', 'fontweight': 'bold', 'fontsize': 12},
    #          horizontalalignment="right")
    # ax = plt.subplot()
    # ax.loglog(Amp[pop1], Eny[pop1], '.', marker='.', markersize=8, color=color_1, label='Pop 1')
    # ax.loglog(Amp[pop2_1], Eny[pop2_1], '.', marker='.', markersize=8, color=color_2, label='Pop 2-1')
    # ax.loglog(Amp[pop2_2], Eny[pop2_2], '.', marker='.', markersize=8, color='orange', label='Pop 2-2')
    # plot_norm(ax, xlabelz[0], xlabelz[2])

    # ------------------------------------------------------------------------------------------------------------------
    # data_1 = pd.DataFrame(
    #     {'Time_1': Time[cls_KKM[0]], 'Eny_1': Eny[cls_KKM[0]], 'Amp_1': Amp[cls_KKM[0]], 'Dur_1': Dur[cls_KKM[0]]})
    # data_2 = pd.DataFrame(
    #     {'Time_2': Time[cls_KKM[1]], 'Eny_2': Eny[cls_KKM[1]], 'Amp_2': Amp[cls_KKM[1]], 'Dur_2': Dur[cls_KKM[1]]})
    # data_1.to_csv(r'', index=None)
    # data_2.to_csv(r'', index=None)

    # ------------------------------------------------------------------------------------------------------------------
    # for trai, title, c in zip([TRAI[pop1], TRAI[pop2_1], TRAI[pop2_both]], ['Pop 1', 'Pop 2_1', 'Pop 2_2'],
    #                           [color_1, color_2, 'orange']):
    #     Res, N = frequency.cal_ave_freq(trai, valid=False, t_lim=50)
    #     frequency.plot_ave_freq(Res, N, title, c)

    # ------------------------------------------------------------------------------------------------------------------
    # idx_1, idx_2 = linear_matching(Amp, Eny, xlabelz[0], xlabelz[2], [2], [-3])
