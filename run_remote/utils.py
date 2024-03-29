import sqlite3
from plot_format import plot_norm
from tqdm import tqdm
import numpy as np
import array
import sys
import math
import os
import multiprocessing
import shutil
import pandas as pd
from scipy.signal import savgol_filter
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from stream_old import *


mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


class Reload:
    def __init__(self, path_pri, path_tra, fold):
        self.path_pri = path_pri
        self.path_tra = path_tra
        self.fold = fold

    def sqlite_read(self, path, mode='vallen'):
        """
        python读取sqlite数据库文件
        """
        mydb = sqlite3.connect(path)  # 链接数据库
        mydb.text_factory = lambda x: str(x, 'gbk', 'ignore')
        cur = mydb.cursor()  # 创建游标cur来执行SQL语句

        # 获取表名
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        Tables = cur.fetchall()  # Tables 为元组列表

        # 获取表结构的所有信息
        if mode == 'vallen':
            if path[-5:] == 'pridb':
                cur.execute("SELECT * FROM {}".format(Tables[3][0]))
                res = cur.fetchall()[-2][1]
            elif path[-5:] == 'tradb':
                cur.execute("SELECT * FROM {}".format(Tables[1][0]))
                res = cur.fetchall()[-3][1]
        elif mode == 'stream':
            cur.execute("SELECT * FROM {}".format(Tables[1][0]))
            res = cur.fetchall()[-1][1]
        return int(res)

    def read_with_time(self, time):
        conn_pri = sqlite3.connect(self.path_pri)
        result_pri = conn_pri.execute(
            "Select SetID, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts, TRAI FROM view_ae_data")
        chan_1, chan_2, chan_3, chan_4 = [], [], [], []
        t = [[] for _ in range(len(time) - 1)]
        N_pri = self.sqlite_read(self.path_pri)
        for _ in tqdm(range(N_pri)):
            i = result_pri.fetchone()
            if i[-2] is not None and i[-2] >= 6 and i[-1] > 0:
                for idx, chan in zip(np.arange(1, 5), [chan_1, chan_2, chan_3, chan_4]):
                    if i[2] == idx:
                        chan.append(i)
                        for j in range(len(t)):
                            if time[j] <= i[1] < time[j + 1]:
                                t[j].append(i)
                                break
                        break
        chan_1 = np.array(chan_1)
        chan_2 = np.array(chan_2)
        chan_3 = np.array(chan_3)
        chan_4 = np.array(chan_4)
        return t, chan_1, chan_2, chan_3, chan_4

    def read_vallen_data(self, lower=2, t_cut=float('inf'), mode='all'):
        data_tra, data_pri, chan_1, chan_2, chan_3, chan_4 = [], [], [], [], [], []
        if mode == 'all' or mode == 'tra only':
            conn_tra = sqlite3.connect(self.path_tra)
            result_tra = conn_tra.execute("Select Time, Chan, Thr, SampleRate, Samples, TR_mV, Data, TRAI FROM view_tr_data")
            N_tra = self.sqlite_read(self.path_tra)
            for _ in tqdm(range(N_tra), ncols=80):
                i = result_tra.fetchone()
                if i[0] > t_cut:
                    break
                data_tra.append(i)
        if mode == 'all' or mode == 'pri only':
            conn_pri = sqlite3.connect(self.path_pri)
            result_pri = conn_pri.execute("Select SetID, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts, TRAI FROM view_ae_data")
            N_pri = self.sqlite_read(self.path_pri)
            for _ in tqdm(range(N_pri), ncols=80):
                i = result_pri.fetchone()
                if i[1] > t_cut:
                    break
                if i[-2] is not None and i[-2] > lower and i[-1] > 0:
                    data_pri.append(i)
                    if i[2] == 1:
                        chan_1.append(i)
                    elif i[2] == 2:
                        chan_2.append(i)
                    elif i[2] == 3:
                        chan_3.append(i)
                    elif i[2] == 4:
                        chan_4.append(i)
        data_tra = sorted(data_tra, key=lambda x: x[-1])
        data_pri = np.array(data_pri)
        chan_1 = np.array(chan_1)
        chan_2 = np.array(chan_2)
        chan_3 = np.array(chan_3)
        chan_4 = np.array(chan_4)
        return data_tra, data_pri, chan_1, chan_2, chan_3, chan_4

    def read_stream_data(self, t_cut=float('inf'), mode='all'):
        data_tra, data_pri, chan_1, chan_2, chan_3, chan_4 = [], [], [], [], [], []
        if mode == 'all' or mode == 'tra only':
            conn_tra = sqlite3.connect(self.path_tra)
            result_tra = conn_tra.execute("Select TRAI, Time, Channel, SampleRate, Samples, TR_μV, Signal FROM data")
            N_tra = self.sqlite_read(self.path_tra, mode='stream')
            for _ in tqdm(range(N_tra), ncols=80):
                i = result_tra.fetchone()
                if i[1] > t_cut:
                    break
                data_tra.append(i)
        if mode == 'all' or mode == 'pri only':
            conn_pri = sqlite3.connect(self.path_pri)
            result_pri = conn_pri.execute("Select TRAI, Time, Channel, Amp, RiseT, Dur, Eny FROM data")
            N_pri = self.sqlite_read(self.path_pri, mode='stream')
            for _ in tqdm(range(N_pri), ncols=80):
                i = result_pri.fetchone()
                if i[1] > t_cut:
                    break
                data_pri.append(i)
                if i[2] == 1:
                    chan_1.append(i)
                elif i[2] == 2:
                    chan_2.append(i)
                elif i[2] == 3:
                    chan_3.append(i)
                elif i[2] == 4:
                    chan_4.append(i)
        data_pri = np.array(data_pri)
        chan_1 = np.array(chan_1)
        chan_2 = np.array(chan_2)
        chan_3 = np.array(chan_3)
        chan_4 = np.array(chan_4)
        return data_tra, data_pri, chan_1, chan_2, chan_3, chan_4

    def read_pac_data(self, path, lower=2):
        os.chdir(path)
        dir_features = os.listdir(path)[0]
        data_tra, data_pri, chan_1, chan_2, chan_3, chan_4 = [], [], [], [], [], []
        with open(dir_features, 'r') as f:
            data_pri = np.array([j.strip(', ') for i in f.readlines()[1:] for j in i.strip("\n")])
        for _ in tqdm(range(N_tra), ncols=80):
            i = result_tra.fetchone()
            data_tra.append(i)
        for _ in tqdm(range(N_pri), ncols=80):
            i = result_pri.fetchone()
            if i[-2] is not None and i[-2] > lower and i[-1] > 0:
                data_pri.append(i)
                if i[2] == 1:
                    chan_1.append(i)
                if i[2] == 2:
                    chan_2.append(i)
                elif i[2] == 3:
                    chan_3.append(i)
                elif i[2] == 4:
                    chan_4.append(i)
        data_tra = sorted(data_tra, key=lambda x: x[-1])
        data_pri = np.array(data_pri)
        chan_1 = np.array(chan_1)
        chan_2 = np.array(chan_2)
        chan_3 = np.array(chan_3)
        chan_4 = np.array(chan_4)
        return data_tra, data_pri, chan_1, chan_2, chan_3, chan_4

    def export_feature(self, t, time):
        for i in range(len(time) - 1):
            with open(self.fold + '-%d-%d.txt' % (time[i], time[i + 1]), 'w') as f:
                f.write('SetID, TRAI, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts\n')
                # ID, Time(s), Chan, Thr(μV), Thr(dB), Amp(μV), Amp(dB), RiseT(s), Dur(s), Eny(aJ), RMS(μV), Counts, Frequency(Hz)
                for i in t[i]:
                    f.write('{}, {}, {:.8f}, {}, {:.7f}, {:.7f}, {:.2f}, {:.2f}, {:.7f}, {:.7f}, {}\n'.format(
                        i[0], i[-1], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9]))


class Export:
    def __init__(self, chan, data_tra, features_path):
        self.data_tra = data_tra
        self.features_path = features_path
        self.chan = chan

    def find_idx(self):
        Res = []
        for i in self.data_tra:
            Res.append(i[-1])
        Res = np.array(Res)
        return Res

    def detect_folder(self):
        tar = './waveform'
        if not os.path.exists(tar):
            os.mkdir(tar)
        else:
            print("=" * 46 + " Warning " + "=" * 45)
            while True:
                ans = input(
                    "The exported data file has been detected. Do you want to overwrite it: (Enter 'yes' or 'no') ")
                if ans.strip() == 'yes':
                    shutil.rmtree(tar)
                    os.mkdir(tar)
                    break
                elif ans.strip() == 'no':
                    sys.exit(0)
                print("Please enter 'yes' or 'no' to continue!")

    def export_waveform(self, chan, thread_id=0, status='normal'):
        if status == 'normal':
            self.detect_folder()
        Res = self.find_idx()
        pbar = tqdm(chan, ncols=80)
        for i in pbar:
            trai = i[-1]
            try:
                j = self.data_tra[int(trai - 1)]
            except IndexError:
                try:
                    idx = np.where(Res == trai)[0][0]
                    j = self.data_tra[idx]
                except IndexError:
                    print('Error 1: TRAI:{} in Channel is not found in data_tra!'.format(trai))
                    continue
            if j[-1] != trai:
                try:
                    idx = np.where(Res == trai)[0][0]
                    j = self.data_tra[idx]
                except IndexError:
                    print('Error 2: TRAI:{} in Channel is not found in data_tra!'.format(trai))
                    continue
            sig = np.multiply(array.array('h', bytes(j[-2])), j[-3] * 1000)
            with open('./waveform/' + self.features_path[:-4] + '_{:.0f}_{:.8f}.txt'.format(trai, j[0]), 'w') as f:
                f.write('Amp(uV)\n')
                for a in sig:
                    f.write('{}\n'.format(a))
            pbar.set_description("Process: %s | Exporting: %s" % (thread_id, int(trai)))

    def accelerate_export(self, N=4):
        # check existing file
        self.detect_folder()

        # Multiprocessing acceleration
        each_core = int(math.ceil(self.chan.shape[0] / float(N)))
        pool = multiprocessing.Pool(processes=N)
        result = []
        for idx, i in enumerate(range(0, self.chan.shape[0], each_core)):
            result.append(pool.apply_async(self.export_waveform, (self.chan[i:i + each_core], idx + 1, 'accelerate',)))

        pool.close()
        pool.join()
        print('Finished export of waveforms!')
        return result


def material_status(component, status):
    if component == 'pure':
        if status == 'random':
            # 0.508, 0.729, 1.022, 1.174, 1.609
            idx_select_2 = [105, 94, 95, 109, 102]
            TRAI_select_2 = [4117396, 4115821, 4115822, 4117632, 4117393]
            # -0.264, -0.022
            idx_select_1 = [95, 60]
            TRAI_select_1 = [124104, 76892]

            idx_same_amp_1 = [45, 62, 39, 41, 56]
            TRAI_same_amp_1 = [88835, 114468, 82239, 84019, 104771]

            idx_same_amp_2 = [61, 118, 139, 91, 136]
            TRAI_same_amp_2 = [74951, 168997, 4114923, 121368, 4078227]

    elif component == 'electrolysis':
        if status == 'random':
            # 0.115, 0.275, 0.297, 0.601, 1.024
            idx_select_2 = [50, 148, 51, 252, 10]
            TRAI_select_2 = [3067, 11644, 3079, 28583, 1501]
            # 0.303, 0.409, 0.534, 0.759, 1.026
            idx_select_1 = [13, 75, 79, 72, 71]
            TRAI_select_1 = [2949, 14166, 14815, 14140, 14090]
        if status == 'amp':
            idx_select_2 = [90, 23, 48, 50, 29]
            TRAI_select_2 = [4619, 2229, 2977, 3014, 2345]

            idx_select_1 = [16, 26, 87, 34, 22]
            TRAI_select_1 = [3932, 7412, 16349, 9001, 6300]
        elif status == 'eny':
            idx_select_2 = [79, 229, 117, 285, 59]
            TRAI_select_2 = [4012, 22499, 7445, 34436, 3282]

            idx_select_1 = [160, 141, 57, 37, 70]
            TRAI_select_1 = [26465, 23930, 11974, 9379, 13667]
    return idx_select_1, idx_select_2, TRAI_select_1, TRAI_select_2


def validation(data_tra, k, filter=False, btype='bandstop'):
    # Time, Amp, RiseTime, Dur, Eny, Counts, TRAI
    i = data_tra[k - 1]
    sig = np.multiply(array.array('h', bytes(i[-2])), i[-3] * 1000)
    time = np.linspace(i[0], i[0] + pow(i[-5], -1) * (i[-4] - 1), i[-4])

    if filter:
        N, CutoffFreq = 4, [550, 650]
        b, a = butter(N, list(map(lambda x: 2 * x * 1e3 / tmp[3], CutoffFreq)), btype)
        sig = filtfilt(b, a, sig)

    thr = i[2]
    valid_wave_idx = np.where(abs(sig) >= thr)[0]
    if not valid_wave_idx.shape[0]:
        return
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
    count = 0
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
    # print(i[0], amplitude, rise_time, duration, energy / pow(10, 4), count, i[-1])
    return i[0], amplitude, rise_time, duration, energy / pow(10, 4), count, i[-1]


def val_TRAI(data_pri, TRAI):
    # Time, Amp, RiseTime, Dur, Eny, Counts, TRAI
    for i in TRAI:
        vallen = data_pri[i - 1]
        print('-' * 80)
        print('{:.8f} {} {} {} {} {:.0f} {:.0f}'.format(vallen[1], vallen[4], vallen[5], vallen[6],
                                                        vallen[-4], vallen[-2], vallen[-1]))
        validation(i - 1)


def save_E_T(Time, Eny, cls_1_KKM, cls_2_KKM, time, displace, smooth_load, strain, smooth_stress):
    df_1 = pd.DataFrame({'time_pop1': Time[cls_KKM[0]], 'energy_pop1': Eny[cls_KKM[0]]})
    df_2 = pd.DataFrame({'time_pop2': Time[cls_KKM[1]], 'energy_pop2': Eny[cls_KKM[1]]})
    df_3 = pd.DataFrame(
        {'time': time, 'displace': displace, 'load': smooth_load, 'strain': strain, 'stress': smooth_stress})
    df_1.to_csv('E-T_electrolysis_pop1.csv')
    df_2.to_csv('E-T_electrolysis_pop2.csv')
    df_3.to_csv('E-T_electrolysis_RawData.csv')


def load_stress(path_curve):
    data = pd.read_csv(path_curve, encoding='gbk').drop(index=[0]).astype('float32')
    data_drop = data.drop_duplicates(['拉伸应变 (应变 1)'])
    time = np.array(data_drop.iloc[:, 0])
    displace = np.array(data_drop.iloc[:, 1])
    load = np.array(data_drop.iloc[:, 2])
    strain = np.array(data_drop.iloc[:, 3])
    stress = np.array(data_drop.iloc[:, 4])
    sort_idx = np.argsort(strain)
    strain = strain[sort_idx]
    stress = stress[sort_idx]
    return time, displace, load, strain, stress


def smooth_curve(time, stress, window_length=99, polyorder=1, epoch=200, curoff=[2500, 25000]):
    y_smooth = savgol_filter(stress, window_length, polyorder, mode='nearest')
    for i in range(epoch):
        if i == 5:
            front = y_smooth
        y_smooth = savgol_filter(y_smooth, window_length, polyorder, mode='nearest')

    front_idx = np.where(time < curoff[0])[0][-1]
    rest_idx = np.where(time > curoff[1])[0][0]
    res = np.concatenate((stress[:40], front[40:front_idx], y_smooth[front_idx:rest_idx], stress[rest_idx:]))
    return res


def filelist_convert(data_path, tar=None):
    file_list = os.listdir(data_path)
    if tar:
        tar += '.txt'
    else:
        tar = data_path.split('/')[-1] + '.txt'
    if tar in file_list:
        exist_idx = np.where(np.array(file_list) == tar)[0][0]
        file_list.pop(exist_idx)
    file_idx = np.array([np.array(i[:-4].split('_')[1:]).astype('int64') for i in file_list])
    return file_list, file_idx


def cal_fitx(tmp, interval_num=1000):
    tmp_min = math.floor(np.log10(min(tmp)))
    tmp_max = math.ceil(np.log10(max(tmp)))
    inter = [i for i in range(tmp_min, tmp_max + 1)]
    for idx in range(1, len(inter)):
        if idx == 1:
            fit_x = np.logspace(inter[idx - 1], inter[idx], interval_num, endpoint=False)
        else:
            fit_x = np.append(fit_x, np.logspace(inter[idx - 1], inter[idx], interval_num, endpoint=False))

    return fit_x


def cal_label(tmp1, tmp2, formula, slope, intercept):
    label = []
    for point in tqdm(zip(tmp1, tmp2)):
        if len(slope) == 2:
            if point[1] >= formula(point[0], slope[0], intercept[0]):
                label.append(0)
            elif point[1] < formula(point[0], slope[1], intercept[1]):
                label.append(2)
            else:
                label.append(1)
        elif len(slope) == 1:
            if point[1] >= formula(point[0], slope[0], intercept[0]):
                label.append(0)
            else:
                label.append(1)
    return label


def linear_matching(tmp1, tmp2, xlabel, ylabel, slope, intercept):
    idx_1, idx_2 = [], []
    formula = lambda x, a, b: pow(x, a) * pow(10, b)
    fit_x = cal_fitx(tmp1, 1000)
    if len(slope) == 1:
        fit_y = [formula(i, slope[0], intercept[0]) for i in fit_x]
        label = cal_label(tmp1, tmp2, formula, slope, intercept)
        idx_1 = np.where(np.array(label) == 0)[0]
        idx_2 = np.where(np.array(label) == 1)[0]

        fig = plt.figure(figsize=[6, 3.9])
        ax = plt.subplot()
        ax.loglog(tmp1[idx_1], tmp2[idx_1], '.', Marker='.', markersize=8, color='red', label='Pop 1')
        ax.loglog(tmp1[idx_2], tmp2[idx_2], '.', Marker='.', markersize=8, color='blue', label='Pop 2')
        ax.loglog(fit_x, fit_y, '.', Marker='.', markersize=0.5, color='black')
        plot_norm(ax, xlabel, ylabel, legend=True)

    elif len(slope) == 2:
        fit_y1 = [formula(i, slope[0], intercept[0]) for i in fit_x]
        fit_y2 = [formula(i, slope[1], intercept[1]) for i in fit_x]
        label = cal_label(tmp1, tmp2, formula, slope, intercept)
        idx_1 = np.where(np.array(label) == 0)[0]
        idx_2 = np.where(np.array(label) == 1)[0]
        idx_3 = np.where(np.array(label) == 2)[0]

        fig = plt.figure(figsize=[6, 3.9])
        ax = plt.subplot()
        ax.loglog(tmp1[idx_1], tmp2[idx_1], '.', Marker='.', markersize=8, color='black', label='Pop 1')
        ax.loglog(tmp1[idx_2], tmp2[idx_2], '.', Marker='.', markersize=8, color='r', label='Pop 2')
        ax.loglog(tmp1[idx_3], tmp2[idx_3], '.', Marker='.', markersize=8, color='b', label='Pop 3')
        ax.loglog(fit_x, fit_y1, '.', Marker='.', markersize=0.5, color='black')
        ax.loglog(fit_x, fit_y2, '.', Marker='.', markersize=0.5, color='black')
        plot_norm(ax, xlabel, ylabel, legend=True)

    else:
        print("Current function don't support fit more than two lines.")

    return idx_1, idx_2


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global max of dmax-chunks of locals max
    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
    # global min of dmin-chunks of locals min
    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]

    return lmin, lmax


def find_nearest(ls, v):
    idx = np.searchsorted(ls, v, side="left")
    if idx > 0 and (idx == len(ls) or abs(v - ls[idx-1]) < abs(v - ls[idx])):
        return idx-1
    else:
        return idx


def stream(file, t_str, t_end, staLen=5, overlap=1, staWin='hamming', IZCRT=0.7, ITU=550, alpha=1.7, t_backNoise=1e4):
    # ====================================================== 数据读取 ======================================================
    with open(file, 'r') as f:
        for _ in range(4):
            f.readline()

        fs = int(f.readline().strip().split()[-1]) * 1e-3
        sig_initial = np.array(list(map(lambda x: float(x.strip()) * 1e4, f.readlines()[4:-1])))
        t_initial = np.array([i / fs for i in range(sig_initial.shape[0])])

    # ====================================================== 计算结果 ======================================================
    t = t_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1] - t_initial[int(t_str // t_initial[1])]
    sig = sig_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1]

    width = int(fs * staLen)
    stride = int(width) - overlap
    t_stE, stE = shortTermEny(sig, width, stride, fs, staWin)
    t_zcR, zcR = zerosCrossingRate(sig, width, stride, fs, staWin)
    stE_dev = cal_deriv(t_stE, stE)
    start, end = find_wave(stE, stE_dev, zcR, t_stE, IZCRT=IZCRT, ITU=ITU, alpha=alpha, t_backNoise=t_backNoise)

    # ====================================================== 图形展示 ======================================================
    x = [t, t_stE, t_stE, t_zcR]
    y = [sig, stE, stE_dev, zcR]
    color = ['black', 'green', 'gray', 'purple']
    ylabel = [r'$Amplitude$ $(μV)$', r'$STEnergy$ $(μV^2 \cdot μs)$', r'$S\dot{T}E$ $(μV^2)$',
              r'$ST\widehat{Z}CR$ $(\%)$']
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
    for idx, ax in enumerate(axes):
        ax.plot(x[idx], y[idx], lw=0.5, color=color[idx])
        if idx == 0:
            for s, e in tqdm(zip(start, end)):
                ax.plot(t[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2],
                        sig[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2], lw=0.5, color='red')
        ax.grid()
        plot_norm(ax, r'$Time$ $(μs)$' if idx == 3 else '', ylabel[idx], legend=False)
    plt.subplots_adjust(wspace=0, hspace=0)


'''
wave_ls = sorted(os.listdir('./wave/txt/'), key=lambda x: int(x.split('-')[0][5:]))
channel = [3]
SampleRate = 2 * pow(10, 7)
tra_data, tra_params = [], []
pri_data, pri_params = [], []
tra_fieldinfo = [tuple(['Time', '[s]', '']), tuple(['SampleRate', '[Hz]', ''])]
tra_globalinfo = [tuple(['TimeBase', SampleRate]), tuple(['BytesPerSample', 2]), tuple(['ValidSets', len(wave_ls)]),
                 tuple(['TRAI', len(wave_ls)])]

pri_fieldinfo = [tuple(['Time', '[s]', '']), tuple(['Amp', '[μV]', '']), tuple(['RiseT', '[μs]', '']), 
                 tuple(['Dur', '[μs]', '']), tuple(['Eny', '[aJ]', ''])]
pri_globalinfo = [tuple(['TimeBase', SampleRate]), tuple(['ValidSets', len(wave_ls)]), 
                  tuple(['TRAI', len(wave_ls)])]

for trai, dir in tqdm(enumerate(wave_ls, 1)):
    ls_tra_tmp, ls_pri_tmp = [], []
    with open('./wave/txt/%s' % dir, 'r') as f:
        w = np.array([list(map(lambda x: float(x), i.strip("\n").split(', '))) for i in f.readlines()[1:]])
    sig = w[:, 1]
    time_label = np.linspace(0, 0 + (1 / 20) * (sig.shape[0] - 1), sig.shape[0])
    max_idx = np.argmax(abs(sig))
    start = time_label[0]
    end = time_label[-1]

    ls_tra_tmp.append(trai)
    ls_tra_tmp.append(float(dir.split('-')[-1][:-4]))
    ls_tra_tmp.append(3)
    ls_tra_tmp.append(SampleRate)
    ls_tra_tmp.append(sig.shape[0])
    ls_tra_tmp.append(1.08023954527964)
    ls_tra_tmp.append((sig / 1.08023954527964).tobytes())

    ls_pri_tmp.append(trai)
    ls_pri_tmp.append(float(dir.split('-')[-1][:-4]))
    ls_pri_tmp.append(3)
    ls_pri_tmp.append(abs(sig[max_idx]))
    ls_pri_tmp.append(time_label[max_idx] - start)
    ls_pri_tmp.append(end - start)
    ls_pri_tmp.append(np.sum(np.multiply(pow(sig, 2), 1 / 20)) / pow(10, 4))

    tra_data.append(tuple(ls_tra_tmp))
    pri_data.append(tuple(ls_pri_tmp))

for id, chan in enumerate(channel, 1):
    tra_params.append(tuple([id, chan, 1.08023954527964]))

for id, chan in enumerate(channel, 1):
    pri_params.append(tuple([id, chan, 1.08023954527964]))

# Create '.tradb' database
con_tra = sqlite3.connect('./test.tradb')
cur_tra = con_tra.cursor()
cur_tra.execute('CREATE TABLE fieldinfo (field TEXT PRIMARY KEY, Unit TEXT, Parameter TEXT)')
cur_tra.executemany("INSERT INTO fieldinfo (field, Unit, Parameter) VALUES(?, ?, ?)", tra_fieldinfo)
cur_tra.execute('CREATE TABLE globalinfo (Key TEXT PRIMARY KEY, Value BLOB)')
cur_tra.executemany("INSERT INTO globalinfo (Key, Value) VALUES(?, ?)", tra_globalinfo)
cur_tra.execute('CREATE TABLE params (ID INTEGER PRIMARY KEY, Channel INTEGER UNIQUE, TR_μV REAL)')
cur_tra.executemany("INSERT INTO params (ID, Channel, TR_μV) VALUES(?, ?, ?)", tra_params)
cur_tra.execute(
    'CREATE TABLE data (TRAI INTEGER PRIMARY KEY, Time REAL, Channel INTEGER, SampleRate INTEGER, Samples INTEGER, TR_μV REAL, Signal TEXT)')
cur_tra.executemany("INSERT INTO data (TRAI, Time, Channel, SampleRate, Samples, TR_μV, Signal) VALUES(?, ?, ?, ?, ?, ?, ?)", tra_data)
con_tra.commit()
con_tra.close()

# Create '.pridb' database
con_pri = sqlite3.connect('./test.pridb')
cur_pri = con_pri.cursor()
cur_pri.execute('CREATE TABLE fieldinfo (field TEXT PRIMARY KEY, Unit TEXT, Parameter TEXT)')
cur_pri.executemany("INSERT INTO fieldinfo (field, Unit, Parameter) VALUES(?, ?, ?)", pri_fieldinfo)
cur_pri.execute('CREATE TABLE globalinfo (Key TEXT PRIMARY KEY, Value BLOB)')
cur_pri.executemany("INSERT INTO globalinfo (Key, Value) VALUES(?, ?)", pri_globalinfo)
cur_pri.execute('CREATE TABLE params (ID INTEGER PRIMARY KEY, Channel INTEGER UNIQUE, TR_μV REAL)')
cur_pri.executemany("INSERT INTO params (ID, Channel, TR_μV) VALUES(?, ?, ?)", pri_params)
cur_pri.execute(
    'CREATE TABLE data (TRAI INTEGER PRIMARY KEY, Time REAL, Channel INTEGER, Amp REAL, RiseT REAL, Dur REAL, Eny REAL)')
cur_pri.executemany("INSERT INTO data (TRAI, Time, Channel, Amp, RiseT, Dur, Eny) VALUES(?, ?, ?, ?, ?, ?, ?)", pri_data)
con_pri.commit()
con_pri.close()
'''

'''
formula = lambda x, a, b:  pow(x, a) * pow(10, b)
fit_x = cal_fitx(Amp, 1000)
fit_y1 = [formula(i, 2, -3) for i in fit_x]
fit_y2 = [formula(i, 2, -2.3) for i in fit_x]
label = cal_label(Amp, Eny, formula, [2, 2], [-2.3, -3])
idx_1 = np.where(np.array(label) == 0)[0]
idx_2 = np.where(np.array(label) == 1)[0]
idx_3 = np.where(np.array(label) == 2)[0]

fig = plt.figure(figsize=[6, 3.9])
fig.text(0.96, 0.2, status, fontdict={'family': 'Arial', 'fontweight': 'bold', 'fontsize': 12},
         horizontalalignment="right")
ax = plt.subplot()
ax.loglog(Amp[idx_1], Eny[idx_1], '.', Marker='.', markersize=3, color='pink', label='Pop 1')
ax.loglog(Amp[idx_2], Eny[idx_2], '.', Marker='.', markersize=3, color='r', label='Pop 2')
ax.loglog(Amp[idx_3], Eny[idx_3], '.', Marker='.', markersize=3, color='b', label='Pop 3')
ax.loglog(fit_x, fit_y1, '.', Marker='.', markersize=0.5, color='black')
ax.loglog(fit_x, fit_y2, '.', Marker='.', markersize=0.5, color='black')
plot_norm(ax, xlabelz[0], xlabelz[2], legend=True)
for k, idx in enumerate([idx_1, idx_2, idx_3], 1):
    with open('%s-pop%d.txt' % (fold, k), 'w') as f:
        f.write('SetID, TRAI, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts\n')
        for i in data_pri[np.where((data_pri[:, 2] == 3) & (data_pri[:, 1] < 5600))[0]][idx]:
            f.write('{:.0f}, {:.0f}, {:.8f}, {:.0f}, {:.7f}, {:.7f}, {:.2f}, {:.2f}, {:.7f}, {:.3f}, {:.0f}\n'.format(
                i[0], i[-1], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9]))
'''

'''
fig = plt.figure(figsize=[6, 3.9])
ax = plt.subplot()
for k in TRAI[cls_KKM[0]]:
    tmp = data_tra[k - 1]
    sig = np.multiply(array.array('h', bytes(tmp[-2])), tmp[-3] * 1000)
    sig = (sig / max(sig))
    time = np.linspace(0, pow(tmp[-5], -1) * (tmp[-4] - 1) * pow(10, 6), tmp[-4])
    if time[-1] < 100:
        continue
    hx = fftpack.hilbert(sig)
    ax.semilogy(time, np.sqrt(sig**2 + hx**2), '.', Marker='.', color=color_1)
for k in TRAI[cls_KKM[1]]:
    tmp = data_tra[k - 1]
    sig = np.multiply(array.array('h', bytes(tmp[-2])), tmp[-3] * 1000)
    sig = (sig / max(sig))
    time = np.linspace(0, pow(tmp[-5], -1) * (tmp[-4] - 1) * pow(10, 6), tmp[-4])
    if time[-1] < 100:
        continue
    hx = fftpack.hilbert(sig)
    ax.semilogy(time, np.sqrt(sig**2 + hx**2), '.', Marker='.', color=color_2)
plot_norm(ax, 'Time (μs)', 'Normalized A$^2$', legend=False)
'''
