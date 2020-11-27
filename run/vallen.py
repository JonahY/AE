# -*- coding: UTF-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm
import csv
import sqlite3
from wave_freq import Waveform, Frequency
from kmeans import KernelKMeans, ICA
from utils import read_data, material_status, validation, val_TRAI, save_E_T
from features import Features


if __name__ == '__main__':
    path = r'E:\data\vallen\Ni-tension test-pure-1-0.01-AE-20201030'
    path_pri = r'Ni-tension test-pure-1-0.01-AE-20201030.pridb'
    path_tra = r'Ni-tension test-pure-1-0.01-AE-20201030.tradb'
    features_path = r'Ni-tension test-pure-1-0.01-AE-20201030.txt'
    os.chdir(path)
    # 316L-1.5-z3-AE-3 sensor-20200530
    # Ni-tension test-electrolysis-1-0.01-AE-20201031
    # Ni-tension test-pure-1-0.01-AE-20201030
    # 2020.11.10-PM-self

    conn_tra = sqlite3.connect(path_tra)
    conn_pri = sqlite3.connect(path_pri)
    result_tra = conn_tra.execute("Select Time, Chan, Thr, SampleRate, Samples, TR_mV, Data, TRAI FROM view_tr_data")
    result_pri = conn_pri.execute(
        "Select SetID, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts, TRAI FROM view_ae_data")

    data_tra, data_pri, chan_2, chan_3, chan_4 = read_data(result_tra, result_pri, path_pri)
    data_tra = sorted(data_tra, key=lambda x: x[-1])
    data_pri = np.array(data_pri)
    chan_2, chan_3, chan_4 = np.array(chan_2), np.array(chan_3), np.array(chan_4)

    # SetID, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts, TRAI
    Time, Amp, RiseT, Dur, Eny, RMS, Counts = chan_2[:, 1], chan_2[:, 4], chan_2[:, 5], \
                                              chan_2[:, 6], chan_2[:, 7], chan_2[:, 8], chan_2[:, 9]

    # SetID, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts, TRAI
    feature_idx = [Amp, Dur, Time]
    xlabelz = ['Amplitude(μV)', 'Duration(μs)', 'Energy(aJ)']
    ylabelz = ['PDF(A)', 'PDF(D)', 'PDF(E)']
    color_1 = [255 / 255, 0 / 255, 102 / 255]  # red
    color_2 = [0 / 255, 136 / 255, 204 / 255]  # blue
    features = Features(color_1, color_2, Time, feature_idx, 8)
    interz, midz = features.cal_interval()

    # ICA and Kernel K-Means
    S_, A_ = ICA(2, Amp, Eny, Dur)
    km = KernelKMeans(n_clusters=2, max_iter=100, random_state=55, verbose=1, kernel="rbf")
    pred = km.fit_predict(S_)
    cls_1_KKM, cls_2_KKM = pred == 1, pred == 0

    # # Select TRAI with material status
    # idx_select_1, idx_select_2, TRAI_select_1, TRAI_select_2 = material_status(path_pri.split('-')[2])

    # Plot log to log scatter and Selected waveform

    features.plot_correlation(Dur, Eny, 'Duration(μs)', 'Energy(aJ)', 'Chan 2', cls_1_KKM, cls_2_KKM)
    features.plot_correlation(Dur, Amp, 'Duration(μs)', 'Amplitude(μV)', 'Chan 2', cls_1_KKM, cls_2_KKM)
    features.plot_correlation(Amp, Eny, 'Amplitude(μV)', 'Energy(aJ)', 'Chan 2', cls_1_KKM, cls_2_KKM)
    features.plot_feature_time(Eny, 'Energy(aJ)')
    # features.cal_waitingTime(Eny, features_path, cls_1_KKM, cls_2_KKM, 'Δt(s)', 'p(Δt)')

    # # Find waves on the edge
    # waveform = Waveform(data_tra, path, path_pri)
    # waveform.find_wave(Dur, Eny, cls_1_KKM, chan_2, [2.0, 2.5], [-1, 0])
    # waveform.save_wave(TRAI_select_1, '1')
    # waveform.save_wave(TRAI_select_2, '2')
    #
    # frequency = Frequency(data_tra, path, path_pri)
    # frequency.plot_wave_frequency(TRAI_select_1, '1')
    # frequency.plot_wave_frequency(TRAI_select_2, '2')

    for i, [idx, inter, mid, xlabel, ylabel] in enumerate(zip(feature_idx, interz, midz, xlabelz, ylabelz)):
        tmp, tmp_1, tmp_2 = sorted(idx), sorted(idx[cls_1_KKM]), sorted(idx[cls_2_KKM])
        features.cal_PDF(tmp, features_path, interz[i], midz[i], tmp_1, tmp_2, xlabel, ylabel)
        features.cal_CCDF(tmp, features_path, tmp_1, tmp_2, xlabel, 'CCD C(s)')
        features.cal_ML(tmp, features_path, tmp_1, tmp_2, xlabel, r'$\epsilon$')
    #
    # features.cal_contour(Eny, Dur, 'Energy(aJ)', 'Duration(μs)', 'Contour')
    # features.plot_correlation(Dur, Eny, 'Duration(μs)', 'Energy(aJ)', 'Chan 2', cls_1_KKM, cls_2_KKM)
    plt.show()
