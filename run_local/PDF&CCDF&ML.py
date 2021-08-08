# -*- coding: UTF-8 -*-
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from matplotlib.pylab import mpl
import warnings

warnings.filterwarnings("ignore")
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def plot_norm(ax, xlabel=None, ylabel=None, zlabel=None, title=None, x_lim=[], y_lim=[], z_lim=[], legend=True,
              grid=False, frameon=True, legend_loc='upper left', font_color='black', legendsize=11, labelsize=14,
              titlesize=15, ticksize=13, linewidth=2, fontname='Arial'):
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)

    # 设置坐标刻度值的大小以及刻度值的字体 Arial, Times New Roman
    ax.tick_params(which='both', width=linewidth, labelsize=ticksize, colors=font_color)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(fontname) for label in labels]

    font_legend = {'family': fontname, 'weight': 'normal', 'size': legendsize}
    font_label = {'family': fontname, 'weight': 'bold', 'size': labelsize, 'color': font_color}
    font_title = {'family': fontname, 'weight': 'bold', 'size': titlesize, 'color': font_color}

    if x_lim:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])
    if z_lim:
        ax.set_zlim(z_lim[0], z_lim[1])
    if legend:
        ax.legend(loc=legend_loc, prop=font_legend, frameon=frameon)
        # plt.legend(loc=legend_loc, prop=font_legend)
    if grid:
        ax.grid(ls='-.')
    if xlabel:
        ax.set_xlabel(xlabel, font_label)
    if ylabel:
        ax.set_ylabel(ylabel, font_label)
    if zlabel:
        ax.set_zlabel(zlabel, font_label)
    if title:
        ax.set_title(title, font_title)
    plt.tight_layout()


class Features:
    def __init__(self, color_1, color_2, time, feature_idx, status):
        self.color_1 = color_1
        self.color_2 = color_2
        self.time = time
        self.feature_idx = feature_idx
        self.status = status

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
        return xx, yy

    def cal_PDF(self, tmp_origin, tmp_1, tmp_2, xlabel, ylabel, LIM=None, INTERVAL_NUM=None, bin_method='log',
                select=None, FIT=False, COLOR=None, LABEL=None):
        """
        Calculate Probability Density Distribution Function
        :param tmp_origin: Energy/Amplitude/Duration in order of magnitude of original data
        :param tmp_1: Energy/Amplitude/Duration in order of magnitude of population 1
        :param tmp_2: Energy/Amplitude/Duration in order of magnitude of population 2
        :param xlabel: 'Amplitude (μV)', 'Duration (μs)', 'Energy (aJ)'
        :param ylabel: 'PDF (A)', 'PDF (D)', 'PDF (E)'
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
                print(inter, mid)
            elif bin_method == 'log':
                inter = self.__cal_log_interval(tmp)
                xx, yy = self.__cal_log(tmp, inter, num)
                print(inter)
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
        plot_norm(ax, xlabel, ylabel, legend_loc='upper right')

    def cal_CCDF(self, tmp_origin, tmp_1, tmp_2, xlabel, ylabel, LIM=None, select=None, FIT=False, COLOR=None, LABEL=None):
        """
        Calculate Complementary Cumulative Distribution Function
        :param tmp_origin: Energy/Amplitude/Duration in order of magnitude of original data
        :param tmp_1: Energy/Amplitude/Duration in order of magnitude of population 1
        :param tmp_2: Energy/Amplitude/Duration in order of magnitude of population 2
        :param xlabel: 'Amplitude (μV)', 'Duration (μs)', 'Energy (aJ)'
        :param ylabel: 'CCDF (A)', 'CCDF (D)', 'CCDF (E)'
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
        N_origin, N1, N2 = len(tmp_origin), len(tmp_1), len(tmp_2)
        fig = plt.figure(figsize=[6, 3.9], num='CCDF--%s' % xlabel)
        fig.text(0.15, 0.2, self.status, fontdict={'family': 'Arial', 'fontweight': 'bold', 'fontsize': 12})
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
        plot_norm(ax, xlabel, ylabel, legend_loc='upper right')

    def cal_ML(self, tmp_origin, tmp_1, tmp_2, xlabel, ylabel, select=None, COLOR=None, LABEL=None):
        """
        Calculate the maximum likelihood function distribution
        :param tmp_origin: Energy/Amplitude/Duration in order of magnitude of original data
        :param tmp_1: Energy/Amplitude/Duration in order of magnitude of population 1
        :param tmp_2: Energy/Amplitude/Duration in order of magnitude of population 2
        :param xlabel: 'Amplitude (μV)', 'Duration (μs)', 'Energy (aJ)'
        :param ylabel: 'ML (A)', 'ML (D)', 'ML (E)'
        :param select: The category displayed each time the function is called, like:
                       [0, None]: Display the original data, the first population, the second population of calculation results.
                       [1, 2]: Only display the first population of calculation results.
        :param COLOR: Color when drawing with original data, population I and population II respectively
        :param LABEL: Format of legend, default ['Whole', 'Pop 1', 'Pop 2']
        :return:
        """
        if select is None:
            select = [0, 3]
        if LABEL is None:
            LABEL = ['Whole', 'Pop 1', 'Pop 2']
        if COLOR is None:
            COLOR = ['black', [1, 0, 0.4], [0, 0.53, 0.8]]
        N_origin, N1, N2 = len(tmp_origin), len(tmp_1), len(tmp_2)
        # fig = plt.figure(figsize=[6, 3.9], num='ML--%s' % xlabel)
        fig = plt.figure(figsize=[6, 3.9])
        fig.text(0.96, 0.2, self.status, fontdict={'family': 'Arial', 'fontweight': 'bold', 'fontsize': 12},
                 horizontalalignment="right")
        ax = plt.subplot()
        ax.set_xscale("log", nonposx='clip')
        TMP, N, LAYER = [tmp_origin, tmp_1, tmp_2], [N_origin, N1, N2], [1, 3, 2]
        for tmp, N, layer, color, label in zip(TMP[select[0]:select[1]], N[select[0]:select[1]],
                                               LAYER[select[0]:select[1]], COLOR[select[0]:select[1]],
                                               LABEL[select[0]:select[1]]):
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
            ax.errorbar(tmp, ML_y, yerr=Error_bar, fmt='o', ecolor=color, color=color, elinewidth=1, capsize=2, ms=3,
                        label=label, zorder=layer)
        plot_norm(ax, xlabel, ylabel, y_lim=[1.3, 2.6], legend_loc='upper right')
