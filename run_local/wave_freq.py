from plot_format import plot_norm
from scipy.fftpack import fft
import array
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from preprocess import Preprocessing
import pywt
import librosa
from librosa import display
import signal_envelope as se
from stream import *
from utils import hl_envelopes_idx
from ssqueezepy import ssq_cwt
from scipy.signal import butter, filtfilt


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
        elif self.device == 'stream':
            # TRAI, Time, Channel, SampleRate, TR_μV, Signal
            sig = np.multiply(array.array('h', bytes(i[-1])), i[-2])
            time = np.linspace(0, pow(i[3], -1) * (i[4] - 1) * pow(10, 6), i[4])

        return time, sig

    @staticmethod
    def find_wave(Dur, Eny, cls_KKM, chan, dur_lim, eny_lim):
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

    def plot_wave_TRAI(self, k, valid=False, color='blue'):
        # Waveform with specific TRAI
        i = self.data_tra[k - 1]
        if i[0 if self.device == 'stream' else -1] != k:
            return str('Error: TRAI %d in data_tra is inconsistent with %d by input!' %
                       (i[0 if self.device == 'stream' else -1], k))
        time, sig = self.cal_wave(i, valid=valid)
        for tmp_tail, s in enumerate(sig[::-1]):
            if s != 0:
                tail = -tmp_tail if tmp_tail > 0 else None
                break
        time, sig = time[:tail], sig[:tail]

        fig = plt.figure(figsize=(6, 4.1), num='Waveform--TRAI %d (%s)' % (k, valid))
        fig.text(0.95, 0.17, self.status, fontdict={'family': 'Arial', 'fontweight': 'bold', 'fontsize': 12},
                 horizontalalignment="right")
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(time, sig, lw=1, color=color)
        if self.device == 'vallen':
            plt.axhline(abs(i[2]), 0, sig.shape[0], linewidth=1, color="black")
            plt.axhline(-abs(i[2]), 0, sig.shape[0], linewidth=1, color="black")
        elif self.device == 'pac':
            plt.axhline(abs(self.thr), 0, sig.shape[0], linewidth=1, color="black")
            plt.axhline(-abs(self.thr), 0, sig.shape[0], linewidth=1, color="black")
        plot_norm(ax, 'Time (μs)', 'Amplitude (μV)', title='TRAI:%d' % k, legend=False, grid=True)

    def plot_stream(self, k, staLen=3, overlap=1, staWin='hamming', IZCRT=0.3, ITU=150, alpha=1, t_backNoise=0,
                    plot=True, classify=False, valid=False, t_str=0, t_end=float('inf')):
        """
        Waveform stream segmentation program
        :param k:
        :param staLen: The duration of the window function, in μs
                       The width of the window function = sampling frequency (MHz) * duration,
                       so this value reflects the duration of the window function on the microsecond scale.
        :param overlap: The overlap coefficient of the window function,
                        stride = the width of the window function - the overlap coefficient
        :param staWin: The name of window function，{'hamming', 'hanning', 'blackman', 'bartlett'}
        :param IZCRT: Identification Zero Crossing Threshold
        :param ITU: Identification Threshold Upper
        :param alpha: weighted factor for the standard deviation of the STE signal, where this a factor can be estimated
                      along with the previous calibration for the thresholds, if a heavy background noise is expected
                      the value for this weighting value must be incremented.
        :param t_backNoise: Used to evaluate background noise time
        :param plot: Whether to draw a figure to show
        :param classify: Whether to display the segmentation results in the figure
        :param valid: Whether to pre-cut the waveform stream according to the threshold
        :param t_str: Waveform stream data segmentation start time
        :param t_end: Waveform stream data segmentation cut-off time
        :return:
        """
        tmp = self.data_tra[int(k - 1)]
        if tmp[0 if self.device == 'stream' else -1] != k:
            return str('Error: TRAI %d in data_tra is inconsistent with %d by input!' %
                       (tmp[0 if self.device == 'stream' else -1], k))
        time, sig = self.cal_wave(tmp, valid=valid)
        if t_str:
            range_idx = np.where((time >= t_str) & (time < t_end))[0]
            time = time[range_idx] - time[range_idx[0]]
            sig = sig[range_idx]

        width = int(tmp[3] * pow(10, -6) * staLen)
        stride = int(width) - overlap
        t_stE, stE = shortTermEny(sig, width, stride, tmp[3] * pow(10, -6), staWin)
        t_zcR, zcR = zerosCrossingRate(sig, width, stride, tmp[3] * pow(10, -6), staWin)
        stE_dev = cal_deriv(t_stE, stE)
        start, end = find_wave(stE, stE_dev, zcR, t_stE, IZCRT=IZCRT, ITU=ITU, alpha=alpha, t_backNoise=t_backNoise)
        if plot:
            x = [time, t_stE, t_stE, t_zcR]
            y = [sig, stE, stE_dev, zcR]
            color = ['black', 'green', 'gray', 'purple']
            ylabel = [r'$Amplitude$ $(μV)$', r'$STEnergy$ $(μV^2 \cdot μs)$', r'$S\dot{T}E$ $(μV^2)$',
                      r'$ST\widehat{Z}CR$ $(\%)$']
            fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
            for idx, ax in enumerate(axes):
                ax.plot(x[idx], y[idx], color=color[idx])
                if classify:
                    if idx == 0:
                        for s, e in zip(start, end):
                            ax.plot(time[np.where((time >= t_stE[s]) & (time <= t_stE[e]))[0]],
                                    sig[np.where((time >= t_stE[s]) & (time <= t_stE[e]))[0]], lw=1, color='red')
                ax.grid()
                plot_norm(ax, r'$Time$ $(μs)$', ylabel[idx], legend=False)
        return start, end, time, sig, t_stE

    def plot_envelope(self, TRAI, COLOR, features_path, valid=False, method='hl', xlog=False):
        fig = plt.figure(figsize=[6, 3.9])
        fig.text(0.95, 0.17, self.status, fontdict={'family': 'Arial', 'fontweight': 'bold', 'fontsize': 12},
                 horizontalalignment="right")
        ax = plt.subplot()
        for idx, [trai, color] in enumerate(zip(TRAI, COLOR)):
            XX, YY = [], []
            for k in tqdm(trai):
                tmp = self.data_tra[k - 1]
                time, sig = self.cal_wave(tmp, valid=valid)
                if time[-1] < 50:
                    continue
                if method == 'se':
                    sig = (sig / max(sig))
                    X_pos_frontier, X_neg_frontier = se.get_frontiers(sig, 0)
                    XX.extend(np.linspace(0, time[-1], len(X_pos_frontier) - 2))
                    YY.extend(sig[X_pos_frontier[2:]] ** 2)
                    if not xlog:
                        ax.semilogy(np.linspace(0, time[-1], len(X_pos_frontier) - 2), sig[X_pos_frontier[2:]] ** 2,
                                    '.', Marker='.', color=color)
                    else:
                        ax.loglog(np.linspace(0, time[-1], len(X_pos_frontier) - 2), sig[X_pos_frontier[2:]] ** 2,
                                  '.', Marker='.', color=color)
                else:
                    sig = sig ** 2 / max(sig ** 2)
                    high_idx, low_idx = hl_envelopes_idx(sig, dmin=60, dmax=60)
                    XX.extend(time[low_idx])
                    YY.extend(sig[low_idx])
                    if not xlog:
                        ax.semilogy(time[low_idx], sig[low_idx], '.', Marker='.', color=color)
                    else:
                        ax.loglog(time[low_idx], sig[low_idx], '.', Marker='.', color=color)

            with open('%s_Decay Function_Pop %d.txt' % (features_path[:-4], idx + 1), 'w') as f:
                f.write('Time (μs), Normalized A$^2$\n')
                for j in range(len(XX)):
                    f.write('{}, {}\n'.format(XX[j], YY[j]))

        plot_norm(ax, 'Time (μs)', 'Normalized A$^2$', legend=False)

    def plot_filtering(self, TRAI, N, CutoffFreq, btype, valid=False, originWave=False, filteredWave=True):
        """
        Signal Filtering
        :param TRAI:
        :param N: Order of filter
        :param CutoffFreq: Cutoff frequency, Wn = 2 * cutoff frequency / sampling frequency, len(Wn) = 2 if btype in ['bandpass', 'bandstop'] else 1
        :param btype: Filter Types, {'lowpass', 'highpass', 'bandpass', 'bandstop'}
        :param originWave: Whether to display the original waveform
        :param filteredWave: Whether to display the filtered waveform
        :param valid: Whether to truncate the waveform according to the threshold
        :return:
        """
        tmp = self.data_tra[int(TRAI - 1)]
        if TRAI != tmp[-1]:
            print('Error: TRAI is incorrect!')
        time, sig = self.cal_wave(tmp, valid=valid)
        b, a = butter(N, list(map(lambda x: 2 * x * 1e3 / tmp[3], CutoffFreq)), btype)
        sig_filter = filtfilt(b, a, sig)

        if originWave:
            fig = plt.figure(figsize=(9.2, 3))
            ax = fig.add_subplot(1, 2, 1)
            ax.plot(time, sig, lw=1, color='blue')
            plot_norm(ax, 'Time (μs)', 'Amplitude (μV)', title='TRAI: %d' % TRAI, legend=False, grid=True)
            ax = fig.add_subplot(1, 2, 2)
            Twxo, Wxo, ssq_freqs, *_ = ssq_cwt(sig, wavelet='morlet', scales='log-piecewise', fs=tmp[3], t=time)
            plt.contourf(time, ssq_freqs * 1000, abs(Twxo), cmap='jet')
            plot_norm(ax, r'Time (μs)', r'Frequency (kHz)', y_lim=[min(ssq_freqs * 1000), 1000], legend=False)

        if filteredWave:
            if btype in ['lowpass', 'highpass']:
                label = 'Frequency %s %d kHz' % ('<' if btype == 'lowpass' else '>', CutoffFreq)
            elif btype == 'bandpass':
                label = '%d kHz < Frequency < %d kHz' % (CutoffFreq[0], CutoffFreq[1])
            else:
                label = 'Frequency < %d kHz or > %d kHz' % (CutoffFreq[0], CutoffFreq[1])
            fig = plt.figure(figsize=(9.2, 3))
            ax = fig.add_subplot(1, 2, 1)
            ax.plot(time, sig_filter, lw=1, color='gray', label=label)
            plot_norm(ax, 'Time (μs)', 'Amplitude (μV)', title='TRAI: %d (%s)' % (TRAI, btype), grid=True,
                      frameon=False, legend_loc='upper right')
            ax = fig.add_subplot(1, 2, 2)
            Twxo, Wxo, ssq_freqs, *_ = ssq_cwt(sig_filter, wavelet='morlet', scales='log-piecewise', fs=tmp[3], t=time)
            plt.contourf(time, ssq_freqs * 1000, abs(Twxo), cmap='jet')
            plot_norm(ax, r'Time (μs)', r'Frequency (kHz)', y_lim=[min(ssq_freqs * 1000), 1000], legend=False)

        return sig_filter

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
    def __init__(self, color_1, color_2, data_tra, path, path_pri, status, device, thr_dB=25, size=250):
        self.data_tra = data_tra
        self.color_1 = color_1
        self.color_2 = color_2
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
            time = np.linspace(0, pow(i[-5], -1) * (i[-4] - 1) * pow(10, 6), i[-4])
            thr, Fs = i[2], i[3]
            # Ts = 1 / Fs
            if valid:
                valid_wave_idx = np.where(abs(sig) >= thr)[0]
                sig = sig[valid_wave_idx[0]:(valid_wave_idx[-1] + 1)]
        elif self.device == 'pac':
            i = self.data_tra[k]
            Fs = 1 / i[2]
            sig = i[-2]
            time = np.linspace(0, i[2] * (i[-3] - 1) * pow(10, 6), i[-3])
            if valid:
                valid_wave_idx = np.where(abs(sig) >= self.thr)[0]
                sig = sig[valid_wave_idx[0]:(valid_wave_idx[-1] + 1)]
        elif self.device == 'stream':
            i = self.data_tra[k]
            sig = np.multiply(array.array('h', bytes(i[-1])), i[-2])
            time = np.linspace(0, pow(i[3], -1) * (i[4] - 1) * pow(10, 6), i[4])
            Fs = i[3]
        N = sig.shape[0]
        fft_y = fft(sig)
        abs_y = np.abs(fft_y)
        normalization = abs_y / N
        normalization_half = normalization[range(int(N / 2))]
        frq = (np.arange(N) / N) * Fs
        half_frq = frq[range(int(N / 2))]
        return half_frq, normalization_half, time

    def cal_ave_freq(self, TRAI, valid=False, t_lim=100):
        Res = np.array([0 for _ in range(self.size)]).astype('float64')
        num = 0
        for j in TRAI:
            half_frq, normalization_half, time = self.cal_frequency(j - 1, valid=valid)
            if time[-1] < t_lim:
                continue
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
            num += 1
        return Res, num

    def cal_wtpacket(self, signal, w, n, plot=False):
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

    def plot_wave_frequency(self, TRAI, valid=False, t_lim=100, n=3, wtpacket=False, wtpacket_eng=False):
        i = self.data_tra[TRAI - 1]
        valid_time, valid_data = self.waveform.cal_wave(i, valid=valid)
        half_frq, normalization_half, time = self.cal_frequency(TRAI - 1, valid=valid)
        if time[-1] < t_lim:
            return

        fig = plt.figure(figsize=(9.2, 3), num='Waveform & Frequency--TRAI %d' % TRAI)
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(valid_time, valid_data, lw=1)
        if self.device != 'stream':
            ax.axhline(abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="black")
            ax.axhline(-abs(i[2]), 0, valid_data.shape[0], linewidth=1, color="black")
        plot_norm(ax, 'Time (μs)', 'Amplitude (μV)', legend=False, grid=True)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(half_frq / 1000, normalization_half, lw=1)
        plot_norm(ax, 'Freq (kHz)', '|Y(freq)|', x_lim=[0, pow(10, 3)], legend=False)

        if wtpacket:
            fig = plt.figure(figsize=(15, 7), num='WaveletPacket--TRAI %d' % TRAI)
            map, wp, energy = self.cal_wtpacket(valid_data, 'db8', n)
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

    def plot_ave_freq(self, Res, N, title, color='blue', y_lim=[0, 1.7], label='whole'):
        fig = plt.figure(figsize=(6, 4.1), num='Average Frequency--%s' % title)
        ax = fig.add_subplot()
        ax.plot(self.grid / 1000, Res / N, lw=1, color=color, label=label)
        plot_norm(ax, x_lim=[0, 800], y_lim=y_lim, xlabel='Freq (kHz)', ylabel='|Y(freq)|', title='Average Frequency')

    def plot_freq_TRAI(self, k, valid=False, color='blue'):
        # Frequency with specific TRAI
        half_frq, normalization_half, _ = self.cal_frequency(k - 1, valid=valid)

        fig = plt.figure(figsize=(6, 4.1), num='Frequency--TRAI:%d (%s)' % (k, valid))
        ax = plt.subplot()
        ax.plot(half_frq / 1000, normalization_half, lw=1, color=color)
        plot_norm(ax, 'Freq (kHz)', '|Y(freq)|', x_lim=[0, pow(10, 3)], title='TRAI:%d' % k, legend=False)

    def plot_2cls_freq(self, TRAI_1, TRAI_2, same):
        fig = plt.figure(figsize=(6.5, 10), num='Frequency with same %s' % same)
        for idx, k in enumerate(TRAI_1):
            half_frq, normalization_half, _ = self.cal_frequency(k - 1)
            ax = fig.add_subplot(5, 2, 1 + idx * 2)
            ax.plot(half_frq / 1000, normalization_half, lw=1)
            plot_norm(ax, 'Freq (kHz)', '|Y(freq)|', x_lim=[0, pow(10, 3)], legend=False)

            half_frq, normalization_half, _ = self.cal_frequency(TRAI_2[idx] - 1)
            ax2 = fig.add_subplot(5, 2, 2 + idx * 2)
            ax2.plot(half_frq, normalization_half, lw=1)
            plot_norm(ax2, 'Freq (Hz)', '|Y(freq)|', x_lim=[0, pow(10, 6)], legend=False)

    def cal_freq_max(self, ALL_TRAI, t_lim=0, status='peak', value=[300000, 500000]):
        freq, stage_idx = [], []
        for idx, trai in enumerate(tqdm(ALL_TRAI)):
            half_frq, normalization_half, time = self.cal_frequency(trai - 1, valid=False)
            if time[-1] < t_lim:
                continue
            if status == 'peak':
                freq.append(half_frq[np.argmax(normalization_half)])
                stage_idx.append(idx)
            elif status == 'three peaks':
                freq_max = 0
                idx_1 = np.where(half_frq < value[0])[0]
                idx_2 = np.where((half_frq >= value[0]) & (half_frq < value[1]))[0]
                idx_3 = np.where(half_frq >= value[1])[0]
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
        return freq / 1000, stage_idx

    def plot_tf_stft(self, TRAI, hop_length=128, save_path=None):
        i = self.data_tra[int(TRAI - 1)]
        if self.device == 'vallen':
            sig = np.multiply(array.array('h', bytes(i[-2])), i[-3] * 1000)
        elif self.device == 'stream':
            sig = np.multiply(array.array('h', bytes(i[-1])), i[-2])

        D = librosa.stft(sig, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        fig, ax = plt.subplots(figsize=(5.12, 5.12))
        _ = librosa.display.specshow(S_db, sr=i[3], hop_length=hop_length, x_axis='time', y_axis='linear', ax=ax)
        ax.set(title='Now with labeled axes!')
        ax.set_ylim(0, 1000000)
        if save_path:
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(os.path.join(save_path, '%i.jpg' % TRAI), pad_inches=0)

    def plot_tf_cwt(self, TRAI, wavelet_name='morl', save_path=None):
        i = self.data_tra[int(TRAI - 1)]
        if self.device == 'vallen':
            sig = np.multiply(array.array('h', bytes(i[-2])), i[-3] * 1000)
            time = np.linspace(0, pow(i[-5], -1) * (i[-4] - 1) * pow(10, 6), i[-4])
        elif self.device == 'stream':
            sig = np.multiply(array.array('h', bytes(i[-1])), i[-2])
            time = np.linspace(0, pow(i[3], -1) * (i[4] - 1) * pow(10, 6), i[4])

        scales = pywt.central_frequency(wavelet_name) * 1e3 / np.arange(1, 1e3, 1e0)
        [cwtmatr_new, frequencies_new] = pywt.cwt(sig, scales, wavelet_name, 1.0 / i[3])
        plt.figure(figsize=(5.12, 5.12))
        plt.contourf(time, frequencies_new / 1000, abs(cwtmatr_new))
        plt.ylim(20, 1000)
        plt.xlabel('Time (μs)')
        plt.ylabel('Frequency (kHz)')
        if save_path:
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(os.path.join(save_path, '%i.jpg' % TRAI), pad_inches=0)

    def plot_tf_wsst(self, TRAI, save_path=None):
        i = self.data_tra[int(TRAI - 1)]
        if self.device == 'vallen':
            sig = np.multiply(array.array('h', bytes(i[-2])), i[-3] * 1000)
            time = np.linspace(0, pow(i[-5], -1) * (i[-4] - 1) * pow(10, 6), i[-4])
        elif self.device == 'stream':
            sig = np.multiply(array.array('h', bytes(i[-1])), i[-2])
            time = np.linspace(0, pow(i[3], -1) * (i[4] - 1) * pow(10, 6), i[4])
        Twxo, Wxo, ssq_freqs, *_ = ssq_cwt(sig, wavelet='morlet', scales='log-piecewise', fs=i[3], t=time)
        fig = plt.figure(figsize=(5.12, 5.12))
        # plt.imshow(np.abs(Twxo), aspect='auto', vmin=0, vmax=.2, cmap='jet')
        plt.contourf(time, ssq_freqs * 1000, abs(Twxo), cmap='jet')
        plt.ylim(min(ssq_freqs * 1000), 1000)
        plt.xlabel(r'Time (μs)')
        plt.ylabel(r'Frequency (kHz)')
        if save_path:
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(os.path.join(save_path, '%i.jpg' % TRAI), pad_inches=0)

    def plot_XXX_Freq(self, freq, feature, ylabel, marker='o', markersize=10, color='blue'):
        fig = plt.figure(figsize=[6, 3.9])
        ax = fig.add_subplot()
        ax.set_yscale("log", nonposy='clip')
        plt.scatter(freq, feature, marker=marker, s=markersize, color=color)
        plot_norm(ax, 'Peak Frequency (kHz)', ylabel, x_lim=[0, 800], legend=False)

    def plot_freqDomain(self, ALL_TRAI, t_lim=100, lw=1):
        fig = plt.figure(figsize=[6, 3.9], num='Frequency domain')
        z = 0
        ax = fig.add_subplot(111, projection='3d')
        for i in ALL_TRAI:
            half_frq, normalization_half, t = self.cal_frequency(i - 1)
            if t[-1] < t_lim:
                continue
            valid_idx = np.where((half_frq / 1000) < 1000)[0]
            # ax.plot(half_frq[valid_idx] / 1000, [z] * valid_idx.shape[0], normalization_half[valid_idx], lw=lw)
            ax.scatter(half_frq[valid_idx] / 1000, [z] * valid_idx.shape[0], normalization_half[valid_idx])
            z += 1
        plot_norm(ax, 'Freq (kHz)', 'Points', '|Y(freq)|', x_lim=[0, 1000], y_lim=[0, z], legend=False)
