import numpy as np
import time
from time import sleep


# def Windows(width, parameter):
#     function = np.zeros(width)
#     for n in range(width):
#         function[n] = (1 - parameter) - parameter * np.cos((2 * 3.14 * n) / (width - 1))
#     return function


# def framing(audio_1, N, move, hamming):
#     framing = np.zeros(1)
#     for i in tqdm(range(0, len(audio_1), move)):
#         if len(audio_1[i:i + N]) == N:
#             tmp = audio_1[i:i + N] * hamming
#             framing = np.append(framing, tmp)
#         else:
#             tmp = audio_1[i:i + N] * hamming[0:len(audio_1[i:i + N])]
#             framing = np.append(framing, tmp)
#     return framing


# def sgn(a):
#     if a >= 0:
#         return 1
#     else:
#         return -1


# def shortTermEny(audio_1, N, move, fs, hamming):
#     short_power = np.zeros(1)
#     sample_interval = 1 / fs
#     for i in range(0, len(audio_1), move):
#         if len(audio_1[i:i + N]) == N:
#             tmp = np.sum(np.multiply(pow(np.abs(audio_1[i:i + N]), 2), sample_interval)) * hamming
#             short_power = np.append(short_power, tmp)
#         else:
#             tmp = np.sum(np.multiply(pow(np.abs(audio_1[i:i + N]), 2), sample_interval)) * hamming[
#                                                                                            0:len(audio_1[i:i + N])]
#             short_power = np.append(short_power, tmp)
#     return short_power


# def shortTermEny(audio_1, N, move):
#     short_power = np.zeros(1)
#     for i in range(0, len(audio_1), move):
#         if len(audio_1[i:i + N]) == N:
#             tmp = pow(np.abs(audio_1[i:i + N]), 2) * hamming
#             short_power = np.append(short_power, tmp)
#         else:
#             tmp = pow(np.abs(audio_1[i:i + N]), 2) * hamming[0:len(audio_1[i:i + N])]
#             short_power = np.append(short_power, tmp)
#     return short_power


def shortTermEny(signal, framelen, stride, fs, window='hamming'):
    """
    :param signal: raw signal of waveform, unit: μs
    :param framelen: length of per frame, type: int
    :param stride: length of translation per frame
    :param fs: sampling rate per millisecond
    :param window: window's function
    :return: time_stE, stE
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
    res = np.zeros(nf)

    try:
        windows = allWindows[window]
    except:
        print("Please select window's function from: hamming, hanning, blackman and bartlett.")
        return t, res

    for i in range(0, nf):
        b = np.square(frames[i:i + 1][0]) * windows * 1.0 / fs
        res[i] = np.sum(b)
    return t, res


def zerosCrossingRate(signal, framelen, stride, fs, window='hamming'):
    """
    :param signal: raw signal of waveform, unit: μs
    :param framelen: length of per frame, type: int
    :param stride: length of translation per frame
    :param fs: sampling rate per millisecond
    :param window: window's function
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
    res = np.zeros(nf)

    try:
        windows = allWindows[window]
    except:
        print("Please select window's function from: hamming, hanning, blackman and bartlett.")
        return t, res

    for i in range(nf):
        tmp = windows * frames[i:i + 1][0]
        for j in range(framelen - 1):
            if tmp[j] * tmp[j + 1] <= 0:
                res[i] += 1
    return t, res / framelen


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


# def zerosCountRate(audio_1, N, move):
#     count = np.zeros(1)
#     for i in range(0, len(audio_1), move):
#         Calculation = 0
#         if len(audio_1[i:i + N]) == N:
#             for j in range(N):
#                 tmp = 0.5 * (np.abs(sgn(audio_1[i + j]) - sgn(audio_1[i + j - 1])))
#                 Calculation += tmp
#             count = np.append(count, Calculation)
#         else:
#             for j in range(len(audio_1[i:i + N])):
#                 tmp = 0.5 * (np.abs(sgn(audio_1[i + j]) - sgn(audio_1[i + j - 1])))
#                 Calculation += tmp
#             count = np.append(count, Calculation)
#     return count


def find_wave(stE, stE_dev, zcR, t_stE, IZCRT=0.3, ITU=75, alpha=0.5, t_backNoise=0):
    start, end = [], []
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
        print(last_end, ITU_tmp, IZCRT_tmp)
        try:
            start_temp = last_end + np.where(stE[last_end + 1:] >= ITU_tmp)[0][0]
        except IndexError:
            print("\r100%|{}| [{:.2f}<?, ?it/s]".format("█" * int((stE.shape[0] - 1) / 10), time.perf_counter() - t0),
                  end="", flush=True)
            return start, end
        start_true = last_end + np.where(np.array(stE_dev[last_end:start_temp + 1]) <= 0)[0][-1] \
            if np.where(np.array(stE_dev[last_end:start_temp]) <= 0)[0].shape[0] else last_end

        # Auto-adjust threshold
        ITU_tmp = ITU + np.mean(stE[last_end:start_true]) + alpha * np.std(stE[last_end:start_true]) \
            if last_end != start_true else ITU_tmp
        IZCRT_tmp = np.mean(zcR[last_end:start_true]) + alpha * np.std(zcR[last_end:start_true]) \
            if last_end != start_true else IZCRT_tmp
        print(ITU_tmp, IZCRT_tmp)

        for j in range(start_temp + 1, stE.shape[0]):
            if stE[j] < ITU_tmp:
                end_temp = j
                break
        ITL = 0.368 * max(stE[start_true:end_temp+1]) if ITU_tmp > 0.368 * max(stE[start_true:end_temp+1]) else ITU_tmp
        print(end_temp, ITL)

        for k in range(end_temp, stE.shape[0]):
            if ((stE[k] < ITL) & (zcR[k] > IZCRT_tmp)) | (k == stE.shape[0] - 1):
                end_true = k
                break

        if start_true >= end_true:
            print("\r100%|{}| [{:.2f}<?, ?it/s]".format(" " * int((stE.shape[0] - 1) / 10), time.perf_counter() - t0),
                  end="", flush=True)
            return start, end

        last_end = end_true
        start.append(start_true)
        end.append(end_true)

        progressbar = "█" * int(last_end / 10)
        space = " " * (int((stE.shape[0] - 2) / 10) - int(last_end / 10))
        print("\r{:^3.0f}%|{}{}| [{:.2f}<?, ?it/s]".format((last_end / (stE.shape[0] - 1)) * 100, progressbar, space,
                                                           time.perf_counter() - t0), end="", flush=True)

    return start, end


'''
tmp = data_tra[int(167537 - 1)]
sig = np.multiply(array.array('h', bytes(tmp[-2])), tmp[-3] * 1000)
t = np.linspace(0, pow(tmp[-5], -1) * (tmp[-4] - 1) * pow(10, 6), tmp[-4])

staLen, overlap, staWin = 3, 1, 'hamming'
IZCRT, ITU, alpha = 0.7, None, 1

width = int(tmp[3] * pow(10, -6) * staLen)
stride = int(width) - overlap
t_stE, stE = shortTermEny(sig, width, stride, 20)
t_zcR, zcR = zerosCrossingRate(sig, width, stride, 20)
stE_dev = cal_deriv(t_stE, stE)
x = [t, t_stE, t_stE, t_zcR]
y = [sig, stE, stE_dev, zcR]
color = ['b', 'green', 'gray', 'purple']
ylabel = ['$Amplitude$ $(μV)$', '$STEnergy$ $(μV^2 \cdot μs)$', "$S\dot{T}E$ $(μV^2)$", '$ST\widehat{Z}CR$ $(\%)$']
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
for idx, ax in enumerate(axes):
    ax.plot(x[idx], y[idx], color=color[idx])
    ax.grid()
    plot_norm(ax, 'Time (μs)', ylabel[idx], legend=False)

'''

'''
tmp = data_tra[int(167537 - 1)]
sig = np.multiply(array.array('h', bytes(tmp[-2])), tmp[-3] * 1000)
t = np.linspace(0, pow(tmp[-5], -1) * (tmp[-4] - 1) * pow(10, 6), tmp[-4])

staLen, overlap, staWin = 3, 1, 'hamming'
IZCRT, ITU, alpha = 0.7, None, 1

width = int(tmp[3] * pow(10, -6) * staLen)
stride = int(width) - overlap
t_stE, stE = shortTermEny(sig, width, stride, 20)
t_zcR, zcR = zerosCrossingRate(sig, width, stride, 20)
stE_dev = cal_deriv(t_stE, stE)
x = [t, t_stE, t_stE, t_zcR]
y = [sig, stE, stE_dev, zcR]
color = ['b', 'green', 'gray', 'purple']
ylabel = ['$Amplitude$ $(μV)$', '$STEnergy$ $(μV^2 \cdot μs)$', "$S\dot{T}E$ $(μV^2)$", '$ST\widehat{Z}CR$ $(\%)$']
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
for idx, ax in enumerate(axes):
    ax.plot(x[idx], y[idx], color=color[idx])
    if idx == 0:
        for s, e in zip(start, end):
            ax.plot(t[np.where((t >= t_stE[s]) & (t <= t_stE[e]))[0]],
                    sig[np.where((t >= t_stE[s]) & (t <= t_stE[e]))[0]], lw=1, color=color_1)
    ax.grid()
    plot_norm(ax, 'Time (μs)', ylabel[idx], legend=False)

'''
'''
trai = 125521
start, end, time, sig, t_stE = waveform.plot_stream(trai, 3, 0, 'hamming', IZCRT=0.3, alpha=0.5, ITU=100, t_backNoise=4, classify=True, t_str=0)
for sta, ed in zip(start, end):
    print(t_stE[sta], t_stE[ed])

'''
'''
for idx, [sta, ed] in enumerate(zip(start, end)):
    with open('./wave/TRAI %d-%d.txt' % (trai, idx), 'w') as f:
        f.write('Time (μs), Amplitude (μV)\n')
        for t, s in zip(time[np.where((time >= t_stE[sta]) & (time <= t_stE[ed]))[0]], sig[np.where((time >= t_stE[sta]) & (time <= t_stE[ed]))[0]]):
            f.write('%f, %f\n' % (t, s))

'''
'''
wave_ls = sorted(os.listdir('./wave/txt/'), key=lambda x: int(x.split('-')[0][5:]))
amplitude, duration, rise_time, energy, t = [], [], [], [], []
for dir in tqdm(wave_ls):
    with open('./wave/txt/%s' % dir, 'r') as f:
        w = np.array([list(map(lambda x: float(x), i.strip("\n").split(', '))) for i in f.readlines()[1:]])
        sig = w[:, 1]
        time_label = np.linspace(0, 0 + (1 / 20) * (sig.shape[0] - 1), sig.shape[0])
        max_idx = np.argmax(abs(sig))
        start = time_label[0]
        end = time_label[-1]
        
        t.append(float(dir.split('-')[-1][:-4]))
        amplitude.append(abs(sig[max_idx]))
        duration.append(end - start)
        rise_time.append(time_label[max_idx] - start)
        energy.append(np.sum(np.multiply(pow(sig, 2), 1 / 20)) / pow(10, 4))
t = np.array(t)
amplitude = np.array(amplitude)
duration = np.array(duration)
rise_time = np.array(rise_time)
energy = np.array(energy)
'''