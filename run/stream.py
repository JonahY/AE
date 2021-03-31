import numpy as np


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



'''
tmp = data_tra[int(6200 - 1)]
sig = np.multiply(array.array('h', bytes(tmp[-2])), tmp[-3] * 1000)
time = np.linspace(0, pow(tmp[-5], -1) * (tmp[-4] - 1) * pow(10, 6), tmp[-4])

staLen, overlap, staWin = 2, 1, 'hamming'
IZCRT, ITU, alpha = 0.7, None, 1

width = int(tmp[3] * pow(10, -6) * staLen)
stride = int(width) - overlap
t_stE, stE = shortTermEny(sig, width, stride, 20)
t_zcR, zcR = zerosCrossingRate(sig, width, stride, 20)
stE_dev = cal_deriv(t_stE, stE)
x = [time, t_stE, t_stE, t_zcR]
y = [sig, stE, stE_dev, zcR]
color = ['b', 'green', 'gray', 'purple']
ylabel = ['$Amplitude$ $(μV)$', '$STEnergy$ $(μV^2 \cdot μs)$', "$S\dot{T}E$ $(μV^2)$", '$ST\widehat{Z}CR$ $(\%)$']
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
for idx, ax in enumerate(axes):
    ax.plot(x[idx], y[idx], color=color[idx])
    ax.grid()
    plot_norm(ax, 'Time (μs)', ylabel[idx], legend=False)
'''