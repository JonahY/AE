from plot_format import plot_norm
from scipy.fftpack import fft
import array
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


class Waveform:
    def __init__(self, color_1, color_2, data_tra, path, path_pri):
        self.data_tra = data_tra
        self.path = path
        self.path_pri = path_pri
        self.color_1 = color_1
        self.color_2 = color_2

    def cal_wave(self, i, valid=True):
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
        return time, sig

    def find_wave(self, Dur, Eny, cls_KKM, chan, dur_lim, eny_lim):
        for i in np.where((np.log10(Dur)[cls_KKM] > dur_lim[0]) & (np.log10(Dur)[cls_KKM] < dur_lim[1]) &
                          (np.log10(Eny)[cls_KKM] > eny_lim[0]) & (np.log10(Eny)[cls_KKM] < eny_lim[1]))[0]:
            # Idx, Dur, Eny, TRAI
            print(i, np.log10(Dur)[cls_KKM][i], np.log10(Eny)[cls_KKM][i], '{:.0f}'.format(chan[cls_KKM][i][-1]))

    def plot_2cls_wave(self, TRAI_select_1, TRAI_select_2, same):
        ylim = [35, 60, 80, 150, 250]
        fig = plt.figure(figsize=(6.5, 10), num='Waveforms with same %s' % same)
        for idx, [j, lim] in enumerate(zip(TRAI_select_1, ylim)):
            i = self.data_tra[j - 1]
            if i[-1] != j:
                print('Error: TRAI in data_tra is inconsistent with that by input!')
                break
            valid_time, valid_data = self.cal_wave(i, valid=False)

            ax = fig.add_subplot(5, 2, 1 + idx * 2)
            ax.plot(valid_time, valid_data, color=self.color_1)
            ax.axhline(abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="black")
            ax.axhline(-abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="black")
            plot_norm(ax, 'Time(μs)', 'Amplitude(μV)', y_lim=[-lim, lim], legend=False, grid=True)

            ax2 = fig.add_subplot(5, 2, 2 + idx * 2)
            i = self.data_tra[TRAI_select_2[idx] - 1]
            if i[-1] != TRAI_select_2[idx]:
                print('Error: TRAI in data_tra is inconsistent with that by input!')
                break
            valid_time, valid_data = self.cal_wave(i, valid=False)
            ax2.plot(valid_time, valid_data, color=self.color_2)
            ax2.axhline(abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="black")
            ax2.axhline(-abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="black")
            plot_norm(ax2, 'Time(μs)', 'Amplitude(μV)', y_lim=[-lim, lim], legend=False, grid=True)

    def plot_wave_TRAI(self, k):
        # Waveform with specific TRAI
        i = self.data_tra[k - 1]
        if i[-1] != k:
            return str('Error: TRAI %d in data_tra is inconsistent with %d by input!' % (i[-1], k))
        time, sig = self.cal_wave(i, valid=False)

        fig = plt.figure(figsize=(6, 4.1), num='Waveform--TRAI:%d' % k)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(time, sig)
        plt.axhline(abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="black")
        plt.axhline(-abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="black")
        plot_norm(ax, 'Time(μs)', 'Amplitude(μV)', title='TRAI:%d' % k, legend=False, grid=True)

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
    def __init__(self, color_1, color_2, data_tra, path, path_pri, size=500):
        self.data_tra = data_tra
        self.waveform = Waveform(color_1, color_2, data_tra, path, path_pri)
        self.size = size
        self.grid = np.linspace(0, pow(10, 6), self.size)

    def cal_frequency(self, k, valid=True):
        i = self.data_tra[k]
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

    def plot_wave_frequency(self, TRAI_select, pop):
        fig = plt.figure(figsize=(6.5, 10), num='Waveform & Frequency--pop%s' % pop)
        for idx, j in enumerate(TRAI_select):
            i = self.data_tra[j - 1]
            valid_time, valid_data = self.waveform.cal_wave(i, valid=False)
            half_frq, normalization_half = self.cal_frequency(j - 1, valid=False)

            ax = fig.add_subplot(5, 2, 1 + idx * 2)
            ax.plot(valid_time, valid_data)
            ax.axhline(abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="black")
            ax.axhline(-abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="black")
            plot_norm(ax, 'Time(μs)', 'Amplitude(μV)', legend=False, grid=True)

            ax = fig.add_subplot(5, 2, 2 + idx * 2)
            ax.plot(half_frq, normalization_half)
            plot_norm(ax, 'Freq (Hz)', '|Y(freq)|', x_lim=[0, pow(10, 6)], legend=False)

    def plot_ave_freq(self, Res, N, title):
        fig = plt.figure(figsize=(6, 4.1), num='Average Frequency--%s' % title)
        ax = fig.add_subplot()
        ax.plot(self.grid, Res / N)
        plot_norm(ax, xlabel='Freq (Hz)', ylabel='|Y(freq)|', title='Average Frequency', legend=False)

    def plot_freq_TRAI(self, k):
        # Frequency with specific TRAI
        half_frq, normalization_half = self.cal_frequency(k, valid=False)

        fig = plt.figure(figsize=(6, 4.1), num='Frequency--TRAI:%d' % (k + 1))
        ax = plt.subplot()
        ax.plot(half_frq, normalization_half)
        plot_norm(ax, 'Freq (Hz)', '|Y(freq)|', title='TRAI:%d' % (k + 1), legend=False)
