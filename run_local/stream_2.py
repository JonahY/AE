import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time


def shortTermEny_zerosCrossingRate(signal, framelen, stride, fs, window='hamming'):
    """
    :param signal: raw signal of waveform, unit: μV
    :param framelen: length of per frame, type: int
    :param stride: length of translation per frame
    :param fs: sampling rate per microsecond
    :param window: window's function, e.g., hamming, hanning, blackman, bartlett
    :return: time_zcR, zcR
    """
    if signal.shape[0] <= framelen:
        nf = 1
    else:
        nf = int(np.ceil((1.0 * signal.shape[0] - framelen + stride) / stride))
    pad_length = int((nf - 1) * stride + framelen)
    zeros = np.zeros((pad_length - signal.shape[0],))
    pad_signal = np.concatenate((signal, zeros))
    indices = np.tile(np.arange(0, framelen), (nf, 1)) + np.tile(np.arange(0, nf * stride, stride),
                                                                 (framelen, 1)).T.astype(np.int32)
    frames = pad_signal[indices]
    allWindows = {'hamming': np.hamming(framelen), 'hanning': np.hanning(framelen), 'blackman': np.blackman(framelen),
                  'bartlett': np.bartlett(framelen)}
    t = np.arange(0, nf) * (stride * 1.0 / fs)
    eny, res = np.zeros(nf), np.zeros(nf)

    try:
        windows = allWindows[window]
    except:
        print("Please select window's function from: hamming, hanning, blackman and bartlett.")
        return t, eny, res

    for i in range(nf):
        frame = frames[i:i + 1][0]
        # calculate zeros crossing rate
        tmp = windows * frame
        for j in range(framelen - 1):
            if tmp[j] * tmp[j + 1] <= 0:
                res[i] += 1

        # calculate short term energy
        b = np.square(frame) * windows / fs
        eny[i] = np.sum(b)

    return t, eny, res / framelen


def cal_deriv(x, y):
    diff_x = []
    for i, j in zip(x[0::], x[1::]):
        diff_x.append(j - i)

    diff_y = []
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)

    slopes = []
    for i in range(len(diff_y)):
        slopes.append(diff_y[i] / diff_x[i])

    deriv = []
    for i, j in zip(slopes[0::], slopes[1::]):
        deriv.append((0.5 * (i + j)))
    deriv.insert(0, slopes[0])
    deriv.append(slopes[-1])

    return deriv


def find_wave_multiOutput(stE, stE_dev, zcR, t_stE, IZCRT=0.3, ITU=75, alpha=0.5, t_backNoise=0):
    start, end = [], []
    startTmp, endTmp, ITUTmp, IZCRTTmp, ITLTmp = [], [], [], [], []
    last_end = 0
    t0 = time.perf_counter()

    # Background noise level
    end_backNoise = np.where(t_stE <= t_backNoise)[0][-1]
    ITU_tmp = ITU + np.mean(stE[last_end:end_backNoise]) + alpha * np.std(stE[last_end:end_backNoise]) \
        if last_end != end_backNoise else ITU
    IZCRT_tmp = np.mean(zcR[last_end:end_backNoise]) + alpha * np.std(zcR[last_end:end_backNoise]) \
        if last_end != end_backNoise else IZCRT
    last_end = end_backNoise

    while last_end < stE.shape[0] - 2:
        print('\nStart to find waveform...\nLast End: %d, ITU: %f, IZCRT: %f' % (last_end, ITU_tmp, IZCRT_tmp))
        try:
            start_temp = last_end + np.where(stE[last_end + 1:] >= ITU_tmp)[0][0]
            startTmp.append(start_temp)
        except IndexError:
            print("\r100%|{}| [{:.2f}<?, ?it/s]".format("█" * int((stE.shape[0] - 1) / 10), time.perf_counter() - t0),
                  end="", flush=True)
            print('No data with short-term energy greater than the threshold (ITU), the search ends.\n')
            return start, end, startTmp, endTmp, ITUTmp, IZCRTTmp, ITLTmp
        start_true = last_end + np.where(np.array(stE_dev[last_end:start_temp + 1]) <= 0)[0][-1] \
            if np.where(np.array(stE_dev[last_end:start_temp]) <= 0)[0].shape[0] else last_end
        print('Successfully found the starting index! %d' % start_true)

        # Auto-adjust threshold
        ITU_tmp = ITU_tmp + np.mean(stE[last_end:start_true]) + alpha * np.std(stE[last_end:start_true]) \
            if last_end != start_true else ITU_tmp
        IZCRT_tmp = np.mean(zcR[last_end:start_true]) + alpha * np.std(zcR[last_end:start_true]) \
            if last_end != start_true else IZCRT_tmp
        print('Auto-adjust threshold! ITU: %f, IZCRT: %f' % (ITU_tmp, IZCRT_tmp))
        ITUTmp.append(ITU_tmp)
        IZCRTTmp.append(IZCRT_tmp)

        for j in range(start_temp + 1, stE.shape[0]):
            if stE[j] < ITU_tmp:
                end_temp = j
                break
        ITL = 0.368 * max(stE[start_true:end_temp + 1]) if ITU_tmp > 0.368 * max(
            stE[start_true:end_temp + 1]) else ITU_tmp
        endTmp.append(end_temp)
        ITLTmp.append(ITL)
        print('Successfully found the temporary ending index! End: %d, ITL: %f' % (end_temp, ITL))

        for k in range(end_temp, stE.shape[0]):
            if ((stE[k] < ITL) & (zcR[k] > IZCRT_tmp)) | (k == stE.shape[0] - 1):
                end_true = k
                print('Starting Index: %d, Ending Index: %d' % (start_true, end_true))
                break

        if start_true >= end_true:
            print("\r100%|{}| [{:.2f}<?, ?it/s]".format(" " * int((stE.shape[0] - 1) / 10), time.perf_counter() - t0),
                  end="", flush=True)
            return start, end, startTmp, endTmp, ITUTmp, IZCRTTmp, ITLTmp

        last_end = end_true
        start.append(start_true)
        end.append(end_true)

        # progressbar = "█" * int(last_end / 10)
        # space = " " * (int((stE.shape[0] - 2) / 10) - int(last_end / 10))
        # print("\r{:^3.0f}%|{}{}| [{:.2f}<?, ?it/s]".format((last_end / (stE.shape[0] - 1)) * 100, progressbar, space,
        #                                                    time.perf_counter() - t0), end="", flush=True)

    return start, end, startTmp, endTmp, ITUTmp, IZCRTTmp, ITLTmp


def plot_norm(ax, xlabel=None, ylabel=None, zlabel=None, title=None, x_lim=[], y_lim=[], z_lim=[], legend=True,
              grid=False, frameon=True, legend_loc='upper left', font_color='black', legendsize=11, labelsize=14,
              titlesize=15, ticksize=13, linewidth=2, fontname='Arial', legendWeight='normal', labelWeight='bold',
              titleWeight='bold'):
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)

    # 设置坐标刻度值的大小以及刻度值的字体 Arial, Times New Roman
    ax.tick_params(which='both', width=linewidth, labelsize=ticksize, colors=font_color)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(fontname) for label in labels]

    font_legend = {'family': fontname, 'weight': legendWeight, 'size': legendsize}
    font_label = {'family': fontname, 'weight': labelWeight, 'size': labelsize, 'color': font_color}
    font_title = {'family': fontname, 'weight': titleWeight, 'size': titlesize, 'color': font_color}

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


if __name__ == '__main__':
    with open(r'F:\PAC\Pure Ni-tension test--0.01-2-AE Vallen&PAC-20211115\stream\thershold cutting\Pure Ni-tension test-0.01-2-STREAM20211116-010729-134_ch1.txt', 'r') as f:
        for _ in range(4):
            f.readline()
        fs = int(f.readline().strip().split()[-1]) * 1e-3
        for _ in range(2):
            f.readline()
        trigger_time = float(f.readline().strip()[15:])
        sig_initial = np.array(list(map(lambda x: float(x.strip()) * 1e4, f.readlines()[1:-1])))
        t_initial = np.array([i / fs for i in range(len(sig_initial))])

    staLen, overlap, staWin, IZCRT, ITU, alpha, t_backNoise = 2, 0, 'hamming', 0.1, 650, 0.5, 0
    t_str, t_end = 0, 1e7
    t = t_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1] - t_initial[int(t_str // t_initial[1])]
    sig = sig_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1]

    width = int(fs * staLen)
    stride = int(width) - overlap
    t_stE, stE, zcR = shortTermEny_zerosCrossingRate(sig, width, stride, fs, staWin)
    stE_dev = cal_deriv(t_stE, stE)
    start, end, startTmp, endTmp, ITUTmp, IZCRTTmp, ITLTmp = find_wave_multiOutput(stE, stE_dev, zcR, t_stE,
                                                                                   IZCRT=IZCRT,
                                                                                   ITU=ITU, alpha=alpha,
                                                                                   t_backNoise=t_backNoise)

    x = [t, t_stE, t_stE, t_stE]
    y = [sig, stE, stE_dev, zcR]
    color = ['black', 'green', 'gray', 'purple']
    ylabel = [r'$Amplitude$ /μV', r'$STEnergy$ /μV$^2 \cdot$μs', r'$S\dot{T}E$ /μV$^2$', r'$ST\widehat{Z}CR$ /%']
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
    for idx, ax in enumerate(axes):
        ax.plot(x[idx], y[idx], lw=1, color=color[idx])
        if idx == 0:
            for s, e in tqdm(zip(start, end)):
                ax.plot(t[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2],
                        sig[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2], lw=1, color='red')
                print(t[int(t_stE[s] // t[1]) + 1], t[int(t_stE[e] // t[1]) + 1],
                      t[int(t_stE[e] // t[1]) + 1] - t[int(t_stE[s] // t[1]) + 1])
                ax.axvspan(t[int(t_stE[s] // t[1]) + 1], t[int(t_stE[e] // t[1]) + 1],
                           facecolor=[84 / 255, 1, 159 / 255],
                           alpha=0.5)
                ax.axvline(x=t[int(t_stE[s] // t[1]) + 1], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid',
                           lw=1)
                ax.axvline(x=t[int(t_stE[e] // t[1]) + 1], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid',
                           lw=1)
            ax.set_xticks(np.array(range(0, 251, 25)))
        elif idx == 1:
            ax.set_yticks(np.array(range(-1000, 10000, 2500)))
            ax.fill_between(np.linspace(0, 5, 10), -1000, [500] * 10, facecolor='orange', alpha=0.2)
            ax.fill_between(x[idx], -1000, y[idx], facecolor=[95 / 255, 158 / 255, 160 / 255], alpha=0.2)
            ax.plot(np.linspace(0, x[idx][endTmp[0]], 100), [ITUTmp[0]] * 100, color='orange', linestyle='dashed', lw=1)
            for i in range(len(start)):
                ax.axvspan(x[idx][startTmp[i]], x[idx][endTmp[i]], facecolor=[95 / 255, 158 / 255, 160 / 255],
                           alpha=0.2)
                ax.fill_between(x[idx][startTmp[i]:endTmp[i] + 1], -1000, y[idx][startTmp[i]:endTmp[i] + 1],
                                facecolor=[28 / 255, 28 / 255, 28 / 255], alpha=0.5)
                ax.axvline(x=x[idx][startTmp[i]], color='k', linestyle='dashdot', lw=1)
                ax.axvline(x=x[idx][endTmp[i]], color='orange', linestyle='dashed', lw=1)
                ax.axvline(x=x[idx][end[i]], color=[70 / 255, 130 / 255, 180 / 255], linestyle='solid', lw=1)
                if i:
                    ax.plot(np.linspace(x[idx][end[i - 1]], x[idx][endTmp[i]], 100), [ITUTmp[i]] * 100, color='orange',
                            linestyle='dashed', lw=1)
                    ax.fill_between(np.linspace(x[idx][end[i - 1]], x[idx][start[i]], 100), -1000, [500] * 100,
                                    facecolor=[0, 1, 1],
                                    alpha=0.5)
        elif idx == 2:
            ax.set_yticks(np.array(range(-2000, 2500, 1000)))
            for i in range(len(start)):
                ax.axvspan(x[idx][start[i]], x[idx][startTmp[i]], facecolor=[84 / 255, 1, 159 / 255], alpha=0.5)
                ax.axvline(x=x[idx][startTmp[i]], color='k', linestyle='dashdot', lw=1)
        else:
            ax.set_yticks(np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]))
            ax.axvspan(0, 5, facecolor='orange', alpha=0.2)
            for i in range(len(start)):
                ax.axvspan(x[idx][endTmp[i]], x[idx][end[i]], facecolor=[84 / 255, 1, 159 / 255], alpha=0.5)
                ax.axvline(x=x[idx][endTmp[i]], color='orange', linestyle='dashed', lw=1)
                ax.axvline(x=x[idx][end[i]], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
                ax.plot(np.linspace(x[idx][endTmp[i]], x[idx][end[i]], 100), [IZCRTTmp[i]] * 100,
                        color=[70 / 255, 130 / 255, 180 / 255],
                        linestyle='dashed', lw=1)
                if i:
                    ax.axvspan(x[idx][end[i - 1]], x[idx][start[i]], facecolor=[0, 1, 1], alpha=0.5)

        ax.grid(linewidth=0.3)
        plot_norm(ax, r'$Time$ /μs' if idx == 3 else '', ylabel[idx], legend=False, labelWeight='normal')

    plt.subplots_adjust(wspace=0, hspace=0)
