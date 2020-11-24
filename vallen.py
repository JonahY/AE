# -*- coding: UTF-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm
from scipy.fftpack import fft
from matplotlib.pylab import mpl
import csv
import array
import sqlite3
import pprint
from matplotlib.ticker import FuncFormatter
from matplotlib import ticker, cm
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state


def sqlite_read(path):
    """
    python读取sqlite数据库文件
    """
    mydb = sqlite3.connect(path)                # 链接数据库
    mydb.text_factory = lambda x: str(x, 'gbk', 'ignore')
    cur = mydb.cursor()                         # 创建游标cur来执行SQL语句

    # 获取表名
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    Tables = cur.fetchall()                     # Tables 为元组列表

    # 获取表结构的所有信息
    cur.execute("SELECT * FROM {}".format(Tables[3][0]))
    res = cur.fetchall()
    return int(res[-2][1]), int(res[-1][1])


# Time, Amp, RiseTime, Dur, Eny, Counts, TRAI
def validation(k):
    i = data_tra[k]
    sig = np.multiply(array.array('h', bytes(i[-2])), i[-3] * 1000)
    time = np.linspace(i[0], i[0] + pow(i[-5], -1) * (i[-4] - 1), i[-4])

    thr = i[2]
    valid_wave_idx = np.where(abs(sig) >= thr)[0]
    valid_time = time[valid_wave_idx[0]:(valid_wave_idx[-1] + 1)]
    start = time[valid_wave_idx[0]]
    end = time[valid_wave_idx[-1]]
    duration = (end - start) * pow(10, 6)
    max_idx = np.argmax(abs(sig))
    amplitude = max(abs(sig))
    rise_time = (time[max_idx] - start) * pow(10, 6)
    valid_data = sig[valid_wave_idx[0]:(valid_wave_idx[-1] + 1)]
    energy = np.sum(np.multiply(pow(valid_data, 2), pow(10, 6) / i[3]))
    RMS = math.sqrt(energy / duration)
    count, idx = 0, 1
    N = len(valid_data)
    for idx in range(1, N):
        if valid_data[idx - 1] >= thr > valid_data[idx]:
            count += 1
    # while idx < N:
    #     if min(valid_data[idx - 1], valid_data[idx]) <= thr < max((valid_data[idx - 1], valid_data[idx])):
    #         count += 1
    #         idx += 2
    #         continue
    #     idx += 1
    print(i[0], amplitude, rise_time, duration, energy / pow(10, 4), count, i[-1])


class KernelKMeans(BaseEstimator, ClusterMixin):
    """
    Kernel K-means

    Reference
    ---------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.
    """

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-3, random_state=None,
                 kernel="rbf", gamma=None, degree=3, coef0=1,
                 kernel_params=None, verbose=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]

        K = self._get_kernel(X)

        sw = sample_weight if sample_weight else np.ones(n_samples)
        self.sample_weight_ = sw

        rs = check_random_state(self.random_state)
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)

        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)

        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_,
                               update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            # Compute the number of samples whose cluster did not change
            # since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_samples < self.tol:
                if self.verbose:
                    print("Converged at iteration", it + 1)
                break

        self.X_fit_ = X

        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the
        kernel trick."""
        sw = self.sample_weight_

        for j in range(self.n_clusters):
            mask = self.labels_ == j

            if np.sum(mask) == 0:
                raise ValueError("Empty cluster found, try smaller n_cluster.")

            denom = sw[mask].sum()
            denomsq = denom * denom

            if update_within:
                KK = K[mask][:, mask]  # K[mask, mask] does not work.
                dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]

            dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom

    def predict(self, X):
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_,
                           update_within=False)
        return dist.argmin(axis=1)


def cal_wave(i, valid=True):
    # Time, Chan, Thr, SampleRate, Samples, TR_mV, Data, TRAI
    sig = np.multiply(array.array('h', bytes(i[-2])), i[-3] * 1000)
    time = np.linspace(0, pow(i[-5], -1) * (i[-4] - 1) * pow(10, 6), i[-4])
    thr = i[2]
    if valid:
        valid_wave_idx = np.where(abs(sig) >= thr)[0]
        start = time[valid_wave_idx[0]]
        end = time[valid_wave_idx[-1]]
        duration = end - start
        N = valid_wave_idx[-1] + 1 - valid_wave_idx[0]
        sig = sig[valid_wave_idx[0]:(valid_wave_idx[-1] + 1)]
        time = np.linspace(0, duration, sig.shape[0])
    return time, sig


def cal_frequency(k, valid=True):
    i = data_tra[k]
    sig = np.multiply(array.array('h', bytes(i[-2])), i[-3] * 1000)
    thr, Fs = i[2], i[3]
    Ts = 1 / Fs
    if valid:
        valid_wave_idx = np.where(abs(sig) >= thr)[0]
        sig = sig[valid_wave_idx[0]:(valid_wave_idx[-1] + 1)]
    N = sig.shape[0]
    fft_y = fft(sig)
    abs_y = np.abs(fft_y)
    normalization = abs_y / N
    normalization_half = normalization[range(int(N / 2))]
    frq = (np.arange(N) / N) * Fs
    half_frq = frq[range(int(N / 2))]
    # freq_max.append(half_frq[np.argmax(normalization_half)])
#     print(i[-1], half_frq[np.argmax(normalization_half)])
    return half_frq, normalization_half


def read_data(path_pri):
    data_tra, data_pri, chan_2, chan_3, chan_4 = [], [], [], [], []
    N_pri, N_tra = sqlite_read(path_pri)
    for _ in tqdm(range(N_tra), ncols=80):
        i = result_tra.fetchone()
        data_tra.append(i)
    for _ in tqdm(range(N_pri), ncols=80):
        i = result_pri.fetchone()
        if i[-2] is not None and i[-2] > 2:
            data_pri.append(i)
            if i[2] == 2:
                chan_2.append(i)
            elif i[2] == 3:
                chan_3.append(i)
            elif i[2] == 4:
                chan_4.append(i)
    return data_tra, data_pri, chan_2, chan_3, chan_4


def ICA(Amp, Eny):
    x = np.zeros([Amp.shape[0], 2])
    x[:, 0], x[:, 1] = np.log10(Amp), np.log10(Eny)
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_nor = (x - x_mean) / x_std

    ica = FastICA(n_components=2)
    S_ = ica.fit_transform(x_nor)  # 重构信号
    A_ = ica.mixing_  # 获得估计混合后的矩阵
    return S_, A_


def Plot_log_log(Dur, Eny, cls_1_KKM, cls_2_KKM, color_1, color_2, idx_select_1=[], idx_select_2=[]):
    ax = plt.subplot()
    ax.scatter(np.log10(Dur)[cls_2_KKM], np.log10(Eny)[cls_2_KKM], s=25, c=color_2, label='Class 2')
    ax.scatter(np.log10(Dur)[cls_1_KKM], np.log10(Eny)[cls_1_KKM], s=25, c=color_1, label='Class 1')
    if idx_select_1:
        ax.scatter(np.log10(Dur)[cls_1_KKM][idx_select_1], np.log10(Eny)[cls_1_KKM][idx_select_1], s=25, c='y',
                   label='Random selection')
    if idx_select_2:
        ax.scatter(np.log10(Dur)[cls_2_KKM][idx_select_2], np.log10(Eny)[cls_2_KKM][idx_select_2], s=25, c='purple',
                   label='Random selection')
    plot_norm(ax, 'Duration(μs)', 'Energy(aJ)', 'Chan 2')


def Plot_wave_frequency(data_tra, TRAI_select, fig):
    for idx, j in enumerate(TRAI_select):
        i = data_tra[j - 1]
        valid_time, valid_data = cal_wave(i, valid=False)

        ax = fig.add_subplot(5, 2, 1 + idx * 2)
        ax.plot(valid_time, valid_data)
        ax.axhline(abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="red")
        ax.axhline(-abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="red")
        plot_norm(ax, 'Time', 'Signal', title='TRAI:%d' % j, legend=False, grid=True)

        half_frq, normalization_half = cal_frequency(j - 1, valid=False)

        ax = fig.add_subplot(5, 2, 2 + idx * 2)
        ax.plot(half_frq, normalization_half)
        plot_norm(ax, 'Freq (Hz)', '|Y(freq)|', 'TRAI:%d' % j, x_lim=[0, pow(10, 6)], legend=False)


def formatnum(x, pos):
    return '$10^{}$'.format(int(x))


def plot_norm(ax, xlabel, ylabel, title='', grid=False, formatter_x=False, formatter_y=False,
              x_lim=[], y_lim=[], legend=True, legend_loc='upper left'):
    formatter1 = FuncFormatter(formatnum)
    if formatter_x:
        ax.xaxis.set_major_formatter(formatter1)
    if formatter_y:
        ax.yaxis.set_major_formatter(formatter1)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    # 设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=12)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('DejaVu Sans') for label in labels]

    font_legend = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 14}
    font_label = {'family': 'DejaVu Sans', 'weight': 'bold', 'size': 15}
    font_title = {'family': 'DejaVu Sans', 'weight': 'bold', 'size': 18}

    if x_lim:
        plt.xlim(x_lim[0], x_lim[1])
    if y_lim:
        plt.ylim(y_lim[0], y_lim[1])
    if legend:
        plt.legend(loc=legend_loc, prop=font_legend)
    if grid:
        ax.grid(ls='-.')
    ax.set_xlabel(xlabel, font_label)
    ax.set_ylabel(ylabel, font_label)
    ax.set_title(title, font_title)
    plt.tight_layout()


if __name__ == '__main__':
    os.chdir(r'E:\data\vallen\Ni-tension test-pure-1-0.01-AE-20201030')
    path_pri = r'Ni-tension test-pure-1-0.01-AE-20201030.pridb'
    path_tra = r'Ni-tension test-pure-1-0.01-AE-20201030.tradb'
    # 316L-1.5-z3-AE-3 sensor-20200530
    # Ni-tension test-electrolysis-1-0.01-AE-20201031
    # Ni-tension test-pure-1-0.01-AE-20201030
    # 2020.11.10-PM-self

    conn_tra = sqlite3.connect(path_tra)
    conn_pri = sqlite3.connect(path_pri)
    result_tra = conn_tra.execute("Select Time, Chan, Thr, SampleRate, Samples, TR_mV, Data, TRAI FROM view_tr_data")
    result_pri = conn_pri.execute(
        "Select SetID, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts, TRAI FROM view_ae_data")

    data_tra, data_pri, chan_2, chan_3, chan_4 = read_data(path_pri)
    data_tra = sorted(data_tra, key=lambda x: x[-1])
    data_pri = np.array(data_pri)
    chan_2, chan_3, chan_4 = np.array(chan_2), np.array(chan_3), np.array(chan_4)

    color_1 = [255 / 255, 0 / 255, 102 / 255]  # red
    color_2 = [0 / 255, 136 / 255, 204 / 255]  # blue

    # SetID, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts, TRAI
    Time, Amp, RiseT, Dur, Eny, RMS, Counts = chan_2[:, 1], chan_2[:, 4], chan_2[:, 5], \
                                              chan_2[:, 6], chan_2[:, 7], chan_2[:, 8], chan_2[:, 9]

    # ICA
    S_, A_ = ICA(Amp, Eny)

    # Kernel K-Means
    km = KernelKMeans(n_clusters=2, max_iter=100, random_state=55, verbose=1, kernel="rbf")
    pred = km.fit_predict(S_)
    cls_1_KKM, cls_2_KKM = pred == 1, pred == 0

    # # 0.115, 0.275, 0.297, 0.601, 1.024
    # idx_select_2 = [50, 148, 51, 252, 10]
    # TRAI_select_2 = [3067, 11644, 3079, 28583, 1501]
    #
    # # 0.303, 0.409, 0.534, 0.759, 1.026
    # idx_select_1 = [13, 75, 79, 72, 71]
    # TRAI_select_1 = [2949, 14166, 14815, 14140, 14090]

    # Plot log to log scatter
    fig1 = plt.figure(figsize=[6, 4.5])
    Plot_log_log(Dur, Eny, cls_1_KKM, cls_2_KKM, color_1, color_2)

    # # Selected waveform
    # fig2 = plt.figure(figsize=(12, 15))
    # Plot_wave_frequency(data_tra, TRAI_select_1, fig2)
    #
    # fig3 = plt.figure(figsize=(12, 15))
    # Plot_wave_frequency(data_tra, TRAI_select_1, fig3)

    plt.show()
