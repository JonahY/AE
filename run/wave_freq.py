from plot_format import plot_norm
from scipy.fftpack import fft
import array
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from preprocess import Preprocessing
import pywt


class Waveform:
    def __init__(self, color_1, color_2, data_tra, path, path_pri, status, device, thr_dB=25):
        self.data_tra = data_tra
        self.path = path
        self.path_pri = path_pri
        self.color_1 = color_1
        self.color_2 = color_2
        self.status = status
        self.device = device
        self.thr = pow(10, thr_dB / 20)

    def cal_wave(self, i, valid=True):
        if self.device == 'vallen':
            # Time, Chan, Thr, SampleRate, Samples, TR_mV, Data, TRAI
            sig = np.multiply(array.array('h', bytes(i[-2])), i[-3] * 1000)
            time = np.linspace(0, pow(i[-5], -1) * (i[-4] - 1) * pow(10, 6), i[-4])
            thr = i[2]
            if valid:
                valid_wave_idx = np.where(abs(sig) >= thr)[0]
                start = time[valid_wave_idx[0]]
                end = time[valid_wave_idx[-1]]
                duration = end - start
                sig = sig[valid_wave_idx[0]:(valid_wave_idx[-1] + 1)]
                time = np.linspace(0, duration, sig.shape[0])
        elif self.device == 'pac':
            sig = i[-2]
            time = np.linspace(0, i[2] * (i[-3] - 1) * pow(10, 6), i[-3])
            if valid:
                valid_wave_idx = np.where(abs(sig) >= self.thr)[0]
                start = time[valid_wave_idx[0]]
                end = time[valid_wave_idx[-1]]
                duration = end - start
                sig = sig[valid_wave_idx[0]:(valid_wave_idx[-1] + 1)]
                time = np.linspace(0, duration, sig.shape[0])
        return time, sig

    def find_wave(self, Dur, Eny, cls_KKM, chan, dur_lim, eny_lim):
        for i in np.where((np.log10(Dur)[cls_KKM] > dur_lim[0]) & (np.log10(Dur)[cls_KKM] < dur_lim[1]) &
                          (np.log10(Eny)[cls_KKM] > eny_lim[0]) & (np.log10(Eny)[cls_KKM] < eny_lim[1]))[0]:
            # Idx, Dur, Eny, TRAI
            print(i, np.log10(Dur)[cls_KKM][i], np.log10(Eny)[cls_KKM][i], '{:.0f}'.format(chan[cls_KKM][i][-1]))

    def plot_2cls_wave(self, TRAI_select_1, TRAI_select_2, same, value, valid=False):
        fig = plt.figure(figsize=(9.2, 3), num='Waveforms with same %s--%d μV' % (same, value))
        fig.text(0.48, 0.24, self.status, fontdict={'family': 'Arial', 'fontweight': 'bold', 'fontsize': 12},
                 horizontalalignment="right")
        fig.text(0.975, 0.24, self.status, fontdict={'family': 'Arial', 'fontweight': 'bold', 'fontsize': 12},
                 horizontalalignment="right")
        i = self.data_tra[TRAI_select_1 - 1]
        if i[-1] != TRAI_select_1:
            print('Error: TRAI %d in data_tra is inconsistent with %d by input!' % (i[-1], TRAI_select_1))
            return
        valid_time, valid_data = self.cal_wave(i, valid=valid)

        ax = fig.add_subplot(1, 2, 1)
        ax.plot(valid_time, valid_data, lw=0.5, color=self.color_1)
        ax.axhline(abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="black")
        ax.axhline(-abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="black")
        plot_norm(ax, xlabel='Time (μs)', ylabel='Amplitude (μV)', legend=False, grid=True)

        ax2 = fig.add_subplot(1, 2, 2)
        i = self.data_tra[TRAI_select_2 - 1]
        if i[-1] != TRAI_select_2:
            print('Error: TRAI %d in data_tra is inconsistent with %d by input!' % (i[-1], TRAI_select_2))
            return
        valid_time, valid_data = self.cal_wave(i, valid=valid)
        ax2.plot(valid_time, valid_data, lw=0.5, color=self.color_2)
        ax2.axhline(abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="black")
        ax2.axhline(-abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="black")
        plot_norm(ax2, xlabel='Time (μs)', ylabel='Amplitude (μV)', legend=False, grid=True)

    def plot_wave_TRAI(self, k, valid=True):
        # Waveform with specific TRAI
        i = self.data_tra[k - 1]
        if i[-1] != k:
            return str('Error: TRAI %d in data_tra is inconsistent with %d by input!' % (i[-1], k))
        time, sig = self.cal_wave(i, valid=valid)

        fig = plt.figure(figsize=(6, 4.1), num='Waveform--TRAI %d (%s)' % (k, valid))
        fig.text(0.95, 0.17, self.status, fontdict={'family': 'Arial', 'fontweight': 'bold', 'fontsize': 12},
                 horizontalalignment="right")
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(time, sig, lw=1)
        if self.device == 'vallen':
            plt.axhline(abs(i[2]), 0, sig.shape[0], linewidth=1, color="black")
            plt.axhline(-abs(i[2]), 0, sig.shape[0], linewidth=1, color="black")
        elif self.device == 'pac':
            plt.axhline(abs(self.thr), 0, sig.shape[0], linewidth=1, color="black")
            plt.axhline(-abs(self.thr), 0, sig.shape[0], linewidth=1, color="black")
        plot_norm(ax, 'Time (μs)', 'Amplitude (μV)', title='TRAI:%d' % k, legend=False, grid=True)

    def save_wave(self, TRAI, pop):
        # Save waveform
        os.chdir(self.path)
        for idx, j in enumerate(tqdm(TRAI)):
            i = self.data_tra[j - 1]
            valid_time, valid_data = self.cal_wave(i)
            with open(self.path_pri[:-6] + '_pop%s-%d' % (pop, idx + 1) + '.txt', 'w') as f:
                f.write('Time, Signal\n')
                for k in range(valid_data.shape[0]):
                    f.write("{}, {}\n".format(valid_time[k], valid_data[k]))


class Frequency:
    def __init__(self, color_1, color_2, data_tra, path, path_pri, status, device, thr_dB=25, size=500):
        self.data_tra = data_tra
        self.waveform = Waveform(color_1, color_2, data_tra, path, path_pri, status, device, thr_dB)
        self.size = size
        self.grid = np.linspace(0, pow(10, 6), self.size)
        self.status = status
        self.device = device
        self.thr = pow(10, thr_dB / 20)

    def cal_frequency(self, k, valid=True):
        if self.device == 'vallen':
            i = self.data_tra[k]
            sig = np.multiply(array.array('h', bytes(i[-2])), i[-3] * 1000)
            thr, Fs = i[2], i[3]
            # Ts = 1 / Fs
            if valid:
                valid_wave_idx = np.where(abs(sig) >= thr)[0]
                sig = sig[valid_wave_idx[0]:(valid_wave_idx[-1] + 1)]
        elif self.device == 'pac':
            i = self.data_tra[k]
            Fs = 1 / i[2]
            sig = i[-2]
            if valid:
                valid_wave_idx = np.where(abs(sig) >= self.thr)[0]
                sig = sig[valid_wave_idx[0]:(valid_wave_idx[-1] + 1)]
        N = sig.shape[0]
        fft_y = fft(sig)
        abs_y = np.abs(fft_y)
        normalization = abs_y / N
        normalization_half = normalization[range(int(N / 2))]
        frq = (np.arange(N) / N) * Fs
        half_frq = frq[range(int(N / 2))]
        return half_frq, normalization_half

    def cal_ave_freq(self, TRAI):
        Res = np.array([0 for _ in range(self.size)]).astype('float64')

        for j in TRAI:
            half_frq, normalization_half = self.cal_frequency(j - 1, valid=False)
            valid_idx = int((pow(10, 6) / max(half_frq)) * half_frq.shape[0])
            tmp = [0 for _ in range(self.size)]
            i = 1
            for j, k in zip(half_frq[:valid_idx], normalization_half[:valid_idx]):
                while True:
                    if self.grid[i - 1] <= j < self.grid[i]:
                        tmp[i - 1] += k
                        break
                    i += 1
            Res += np.array(tmp)
        return Res

    def cla_wtpacket(self, signal, w, n, plot=False):
        w = pywt.Wavelet(w)
        wp = pywt.WaveletPacket(data=signal, wavelet=w, mode='symmetric', maxlevel=n)

        map = {}
        map[1] = signal
        for row in range(1, n + 1):
            lev = []
            for i in [node.path for node in wp.get_level(row, 'freq')]:
                map[i] = wp[i].data

        re = []
        for i in [node.path for node in wp.get_level(n, 'freq')]:
            re.append(wp[i].data)
        energy = []
        for i in re:
            energy.append(pow(np.linalg.norm(i, ord=None), 2))

        if plot:
            plt.figure(dpi=100)
            plt.subplot(n + 1, 1, 1)
            plt.plot(map[1])
            for i in range(2, n + 2):
                level_num = pow(2, i - 1)
                # ['aaa', 'aad', 'add', 'ada', 'dda', 'ddd', 'dad', 'daa']
                re = [node.path for node in wp.get_level(i - 1, 'freq')]
                for j in range(1, level_num + 1):
                    plt.subplot(n + 1, level_num, level_num * (i - 1) + j)
                    plt.plot(map[re[j - 1]])
            plt.figure(dpi=100)
            values = [i / sum(energy) for i in energy]
            index = np.arange(pow(2, n))
            p2 = plt.bar(index, values, 0.45, label="num", color="#87CEFA")
            plt.xlabel('clusters')
            plt.ylabel('number of reviews')
            plt.title('Cluster Distribution')
            plt.xticks(index, ('7', '8', '9', '10', '11', '12', '13', '14'))
            plt.legend(loc="upper right")
        return map, wp, energy

    def plot_wave_frequency(self, TRAI, valid=False, n=3, wtpacket=False, wtpacket_eng=False):
        fig = plt.figure(figsize=(9.2, 3), num='Waveform & Frequency--TRAI %d' % TRAI)
        i = self.data_tra[TRAI - 1]
        valid_time, valid_data = self.waveform.cal_wave(i, valid=valid)
        half_frq, normalization_half = self.cal_frequency(TRAI - 1, valid=valid)

        ax = fig.add_subplot(1, 2, 1)
        ax.plot(valid_time, valid_data)
        ax.axhline(abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="black")
        ax.axhline(-abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="black")
        plot_norm(ax, 'Time (μs)', 'Amplitude (μV)', legend=False, grid=True)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(half_frq, normalization_half)
        plot_norm(ax, 'Freq (Hz)', '|Y(freq)|', x_lim=[0, pow(10, 6)], legend=False)

        if wtpacket:
            fig = plt.figure(figsize=(15, 7), num='WaveletPacket--TRAI %d' % TRAI)
            map, wp, energy = self.cla_wtpacket(valid_data, 'db8', n)
            for i in range(2, n + 2):
                level_num = pow(2, i - 1)
                # ['aaa', 'aad', 'add', 'ada', 'dda', 'ddd', 'dad', 'daa']
                re = [node.path for node in wp.get_level(i - 1, 'freq')]
                for j in range(1, level_num + 1):
                    ax = fig.add_subplot(n, level_num, level_num * (i - 2) + j)
                    ax.plot(map[re[j - 1]])
                    plot_norm(ax, '', '', legend=False)
            if wtpacket_eng:
                fig = plt.figure(figsize=(4.6, 3), num='WaveletPacket Energy--TRAI %d' % TRAI)
                ax = fig.add_subplot()
                values = [i / sum(energy) for i in energy]
                index = np.arange(pow(2, n))
                ax.bar(index, values, 0.45, color="#87CEFA")
                plot_norm(ax, 'Clusters', 'Reviews (%)', legend=False)

    def plot_ave_freq(self, Res, N, title):
        fig = plt.figure(figsize=(6, 4.1), num='Average Frequency--%s' % title)
        ax = fig.add_subplot()
        ax.plot(self.grid, Res / N)
        plot_norm(ax, xlabel='Freq (Hz)', ylabel='|Y(freq)|', title='Average Frequency', legend=False)

    def plot_freq_TRAI(self, k, valid=False):
        # Frequency with specific TRAI
        half_frq, normalization_half = self.cal_frequency(k - 1, valid=valid)

        fig = plt.figure(figsize=(6, 4.1), num='Frequency--TRAI:%d (%s)' % (k, valid))
        ax = plt.subplot()
        ax.plot(half_frq, normalization_half)
        plot_norm(ax, 'Freq (Hz)', '|Y(freq)|', x_lim=[0, pow(10, 6)], title='TRAI:%d' % k, legend=False)

    def plot_2cls_freq(self, TRAI_1, TRAI_2, same):
        fig = plt.figure(figsize=(6.5, 10), num='Frequency with same %s' % same)
        for idx, k in enumerate(TRAI_1):
            half_frq, normalization_half = self.cal_frequency(k - 1)
            ax = fig.add_subplot(5, 2, 1 + idx * 2)
            ax.plot(half_frq, normalization_half)
            plot_norm(ax, 'Freq (Hz)', '|Y(freq)|', x_lim=[0, pow(10, 6)], legend=False)

            half_frq, normalization_half = self.cal_frequency(TRAI_2[idx] - 1)
            ax2 = fig.add_subplot(5, 2, 2 + idx * 2)
            ax2.plot(half_frq, normalization_half)
            plot_norm(ax2, 'Freq (Hz)', '|Y(freq)|', x_lim=[0, pow(10, 6)], legend=False)

    def cal_freq_max(self, ALL_TRAI, status='peak'):
        freq, stage_idx = [], []
        for trai in tqdm(ALL_TRAI):
            half_frq, normalization_half = self.cal_frequency(trai - 1)
            if status == 'peak':
                freq.append(half_frq[np.argmax(normalization_half)])
            elif status == 'three peaks':
                freq_max = []
                idx_1 = np.where(half_frq < 300000)
                idx_2 = np.where((half_frq >= 300000) & (half_frq < 500000))
                idx_3 = np.where(half_frq >= 500000)
                normalization_max = 0
                if idx_1.shape[0] != 0 and idx_2.shape[0] != 0 and idx_3.shape[0] != 0:
                    for i, idx in enumerate([idx_1, idx_2, idx_3]):
                        if max(normalization_half[idx]) > normalization_max:
                            idx_max = idx
                            tmp = i + 1
                            normalization_max = max(normalization_half[idx])
                            freq_max = half_frq[idx_max][np.argmax(normalization_half[idx_max])]
                freq.append(freq_max)
                stage_idx.append(tmp)
        freq = np.array(freq)
        stage_idx = np.array(stage_idx)
        return freq, stage_idx