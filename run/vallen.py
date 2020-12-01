# -*- coding: UTF-8 -*-
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
from wave_freq import Waveform, Frequency
from kmeans import KernelKMeans, ICA
from utils import *
from features import Features
import multiprocessing
from multiprocessing import cpu_count
import sys


class Done:
    def __init__(self, Res, data_tra, features_path):
        self.Res = Res
        self.data_tra = data_tra
        self.features_path = features_path

    def main(self, chan):
        for i in tqdm(chan, ncols=80):
            trai = i[-1]
            idx = np.where(self.Res == trai)[0][0]
            j = self.data_tra[idx]
            if j[-1] != trai:
                print('Error: TRAI:{} in data_tra is not inconsistent with {} in Channel!'.format(j[-1], trai))
                continue
            sig = np.multiply(array.array('h', bytes(j[-2])), j[-3] * 1000)
            with open('./waveform/' + self.features_path[:-4] + '_{:.0f}_{:.8f}.txt'.format(trai, j[0]), 'w') as f:
                f.write('Amp(uV)\n')
                for a in sig:
                    f.write('{}\n'.format(a))


if __name__ == '__main__':
    path = r'E:\data\vallen'
    fold = 'Ni-tension test-electrolysis-1-0.01-AE-20201031'
    path_pri = fold + '.pridb'
    path_tra = fold + '.tradb'
    features_path = fold + '.txt'
    os.chdir('\\'.join([path, fold]))
    # 316L-1.5-z3-AE-3 sensor-20200530
    # Ni-tension test-electrolysis-1-0.01-AE-20201031
    # Ni-tension test-pure-1-0.01-AE-20201030
    # 2020.11.10-PM-self

    data_tra, data_pri, chan_1, chan_2, chan_3, chan_4 = read_data(path_pri, path_tra, lower=2)
    print('Channel 1: {} | Channel 2: {} | Channel 3: {} | Channel 4: {}'.format(chan_1.shape[0], chan_2.shape[0],
                                                                                 chan_3.shape[0], chan_4.shape[0]))
    # SetID, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts, TRAI
    chan = chan_2
    # Time, Amp, RiseT, Dur, Eny, RMS, Counts = chan[:, 1], chan[:, 4], chan[:, 5], \
    #                                           chan[:, 6], chan[:, 7], chan[:, 8], chan[:, 9]

    # Export waveforms to txt
    export = Export(chan, data_tra, features_path)
    result = export.accelerate_export(cpu_count())  # Use multiprocessing to accelerate exporting
    # export.export_waveform(chan)    # Use one thread to export

    # # SetID, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts, TRAI
    # feature_idx = [Amp, Dur, Time]
    # xlabelz = ['Amplitude(μV)', 'Duration(μs)', 'Energy(aJ)']
    # ylabelz = ['PDF(A)', 'PDF(D)', 'PDF(E)']
    # color_1 = [255 / 255, 0 / 255, 102 / 255]  # red
    # color_2 = [0 / 255, 136 / 255, 204 / 255]  # blue
    # features = Features(color_1, color_2, Time, feature_idx, 8)
    # interz, midz = features.cal_interval()

    # # ICA and Kernel K-Means
    # S_, A_ = ICA(2, Amp, Eny, Dur)
    # km = KernelKMeans(n_clusters=2, max_iter=100, random_state=55, verbose=1, kernel="rbf")
    # pred = km.fit_predict(S_)
    # cls_1_KKM, cls_2_KKM = pred == 0, pred == 1

    # # Select TRAI with material component and status
    # idx_same_amp_1, idx_same_amp_2, TRAI_same_amp_1, TRAI_same_amp_2 = material_status(path_pri.split('-')[2], 'amp')
    # idx_same_eny_1, idx_same_eny_2, TRAI_same_eny_1, TRAI_same_eny_2 = material_status(path_pri.split('-')[2], 'eny')
    # TRAI_1_all = chan[cls_1_KKM][:, -1].astype(int)
    # TRAI_2_all = chan[cls_2_KKM][:, -1].astype(int)
    # TRAI_all = np.append(TRAI_1_all, TRAI_2_all)

    # # Plot log to log scatter and Selected waveform
    # features.plot_correlation(Dur, Eny, 'Duration(μs)', 'Energy(aJ)', 'Chan 2', cls_1_KKM, cls_2_KKM)
    # features.plot_correlation(Dur, Amp, 'Duration(μs)', 'Amplitude(μV)', 'Chan 2', cls_1_KKM, cls_2_KKM)
    # features.plot_correlation(Amp, Eny, 'Amplitude(μV)', 'Energy(aJ)', cls_1_KKM,
    #                           cls_2_KKM, idx_same_amp_1, idx_same_amp_2, title='Same amplitude')
    # features.plot_correlation(Amp, Eny, 'Amplitude(μV)', 'Energy(aJ)', cls_1_KKM,
    #                           cls_2_KKM, idx_same_eny_1, idx_same_eny_2, title='Same energy')
    # features.plot_correlation(Amp, Eny, 'Amplitude(μV)', 'Energy(aJ)', cls_1_KKM, cls_2_KKM)
    # features.plot_feature_time(Eny, 'Energy(aJ)')
    # features.cal_waitingTime(Eny, features_path, cls_1_KKM, cls_2_KKM, 'Δt(s)', 'p(Δt)')

    # Find waves on the edge
    # waveform = Waveform(color_1, color_2, data_tra, path, path_pri)
    # waveform.plot_2cls_wave(TRAI_same_amp_1, TRAI_same_amp_2, 'amplitude')
    # waveform.plot_2cls_wave(TRAI_same_eny_1, TRAI_same_eny_2, 'energy')
    # waveform.find_wave(Dur, Eny, cls_1_KKM, chan_2, [2.0, 2.5], [-1, 0])
    # waveform.save_wave(TRAI_select_1, '1')
    # waveform.save_wave(TRAI_select_2, '2')

    # TRAI_1_all = chan[cls_1_KKM][:, -1].astype(int)
    # TRAI_2_all = chan[cls_2_KKM][:, -1].astype(int)
    # TRAI_all = np.append(TRAI_1_all, TRAI_2_all)
    #
    # frequency = Frequency(color_1, color_2, data_tra, path, path_pri)
    # for trai, title in zip([TRAI_all, TRAI_1_all, TRAI_2_all], ['Whole', 'Population 1', 'Population 2']):
    #     Res = frequency.cal_ave_freq(trai)
    #     frequency.plot_ave_freq(Res, trai.shape[0], title)
    # frequency.plot_wave_frequency(TRAI_select_1, '1')
    # frequency.plot_wave_frequency(TRAI_select_2, '2')

    # for i, [idx, inter, mid, xlabel, ylabel] in enumerate(zip(feature_idx, interz, midz, xlabelz, ylabelz)):
    #     tmp, tmp_1, tmp_2 = sorted(idx), sorted(idx[cls_1_KKM]), sorted(idx[cls_2_KKM])
    #     features.cal_PDF(tmp, features_path, interz[i], midz[i], tmp_1, tmp_2, xlabel, ylabel)
    #     features.cal_CCDF(tmp, features_path, tmp_1, tmp_2, xlabel, 'CCD C(s)')
    #     features.cal_ML(tmp, features_path, tmp_1, tmp_2, xlabel, r'$\epsilon$')

    # features.cal_contour(Eny, Dur, 'Energy(aJ)', 'Duration(μs)', 'Contour')
    # features.plot_correlation(Dur, Eny, 'Duration(μs)', 'Energy(aJ)', 'Chan 2', cls_1_KKM, cls_2_KKM)
    # plt.show()
