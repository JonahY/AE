import numpy as np
import time
from tqdm import tqdm
import sys
import os
import multiprocessing
from multiprocessing import cpu_count
import math
import argparse
import traceback
import scipy.io as scio


def shortTermEny_zerosCrossingRate(signal, framelen, stride, fs, window='hamming'):
    """
    :param signal: raw signal of waveform, unit: μV
    :param framelen: length of per frame, type: int
    :param stride: length of translation per frame
    :param fs: sampling rate per microsecond
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


def find_wave(stE, stE_dev, zcR, t_stE, IZCRT=0.3, ITU=75, alpha=0.5, t_backNoise=0):
    start, end = [], []
    last_end = 0

    # Background noise level
    end_backNoise = np.where(t_stE <= t_backNoise)[0][-1]
    ITU_tmp = ITU + np.mean(stE[last_end:end_backNoise]) + alpha * np.std(stE[last_end:end_backNoise]) \
        if last_end != end_backNoise else ITU
    IZCRT_tmp = np.mean(zcR[last_end:end_backNoise]) + alpha * np.std(zcR[last_end:end_backNoise]) \
        if last_end != end_backNoise else IZCRT
    last_end = end_backNoise

    while last_end < stE.shape[0] - 2:
        try:
            start_temp = last_end + np.where(stE[last_end + 1:] >= ITU_tmp)[0][0]
        except IndexError:
            return start, end
        start_true = last_end + np.where(np.array(stE_dev[last_end:start_temp + 1]) <= 0)[0][-1] \
            if np.where(np.array(stE_dev[last_end:start_temp]) <= 0)[0].shape[0] else last_end

        # Auto-adjust threshold
        ITU_tmp = ITU + np.mean(stE[last_end:start_true]) + alpha * np.std(stE[last_end:start_true]) \
            if start_true - last_end > 10 else ITU_tmp
        IZCRT_tmp = np.mean(zcR[last_end:start_true]) + alpha * np.std(zcR[last_end:start_true]) \
            if start_true - last_end > 10 else IZCRT_tmp

        for j in range(start_temp + 1, stE.shape[0]):
            if stE[j] < ITU_tmp:
                end_temp = j
                break
        ITL = 0.368 * max(stE[start_true:end_temp + 1]) if ITU_tmp > 0.368 * max(
            stE[start_true:end_temp + 1]) else ITU_tmp

        for k in range(end_temp, stE.shape[0]):
            if ((stE[k] < ITL) & (zcR[k] > IZCRT_tmp)) | (k == stE.shape[0] - 1):
                end_true = k
                break

        if start_true >= end_true:
            return start, end

        last_end = end_true
        start.append(start_true)
        end.append(end_true)

    return start, end


def cut_stream(files, streamFold, saveFold, config):
    try:
        pbar = tqdm(files, ncols=100)
        for file in pbar:
            pbar.set_description('File name: %s' % file[43:-4])

            dataMat = scio.loadmat(os.path.join(streamFold, file))
            fs = dataMat['Sampling rate'][0][0]
            trigger_time = dataMat['Trigger time'][0][0]

            if file not in os.listdir(config.featuresFold):
                width = int(fs * config.staLen)
                stride = int(width) - config.overlap
                t_stE, stE, zcR = shortTermEny_zerosCrossingRate(dataMat['Voltage'][0], width, stride, fs, config.staWin)
                stE_dev = cal_deriv(t_stE, stE)

                # save calculated values
                scio.savemat(os.path.join(config.featuresFold, file), {'Trigger time': trigger_time,
                                                                       'StaWin': config.staWin,
                                                                       'StaLen': config.staLen,
                                                                       'Overlap': config.overlap,
                                                                       'Width': width,
                                                                       'Stride': stride,
                                                                       't_stE': t_stE,
                                                                       'stE': stE,
                                                                       'stE_dev': stE_dev,
                                                                       'zcR': zcR})

            else:
                featuresMat = scio.loadmat(os.path.join(config.featuresFold, file))
                t_stE, stE, stE_dev, zcR = featuresMat['t_stE'][0], featuresMat['stE'][0], featuresMat['stE_dev'][0], featuresMat['zcR'][0]

            start, end = find_wave(stE, stE_dev, zcR, t_stE, IZCRT=config.IZCRT, ITU=config.ITU, alpha=config.alpha,
                                   t_backNoise=config.t_backNoise)

            for out, [s, e] in enumerate(zip(start, end), 1):
                with open(os.path.join(saveFold, '{}-{}.txt'.format(file[:-4], out)), 'w') as f:
                    f.write(f'Trigger time of stream file (s)\n{trigger_time:.8f}\n')
                    f.write(f'Trigger time of AE event (μs)\n{(int(t_stE[s] // (1 / fs)) + 1) / fs:.1f}\n\n')
                    f.write('Amplitude (μV)\n')
                    for i in range(int(t_stE[s] // (1 / fs)) + 1, int(t_stE[e] // (1 / fs)) + 2):
                        f.write(f"{dataMat['Voltage'][0][i]}\n")

            with open(os.path.join(saveFold, 'log'), 'a') as f:
                f.write('%s\n' % file)

    except Exception as e:
        print('Error: %s' % e)
        print(traceback.format_exc())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-streamF", "--streamFold", type=str, default='/home/Yuanbincheng/data/stream/threshold',
                        help="Absolute path of .mat folder(add 'r' in front)")
    parser.add_argument("-saveF", "--saveFold", type=str,
                        default=r'/home/Yuanbincheng/data/stream/alpha_1.3/waveforms_500',
                        help="Absolute path of storage folder(add 'r' in front)")
    parser.add_argument("-f", "--first", type=int, default=1, choices=[0, 1],
                        help="Only the [1] is passed in for the first calculation, and only the streaming file that "
                             "appears in the storage location needs to be calculated later.")
    parser.add_argument("-saveFNew", "--saveFoldNew", type=str,
                        default=r'/home/Yuanbincheng/data/stream/alpha_1.3/waveforms',
                        help="Absolute path of new storage folder(add 'r' in front), "
                             "Only used except for the first calculation.")
    parser.add_argument("-featuresF", "--featuresFold", type=str,
                        default=r'/home/Yuanbincheng/data/stream/features',
                        help="Absolute path of new storage folder(add 'r' in front)")
    parser.add_argument("-cpu", "--processor", type=int, default=cpu_count(), help="Number of Threads")
    parser.add_argument("-detect", "--detection", type=int, default=0, choices=[0, 1],
                        help="Whether to detect log file")
    parser.add_argument("-sL", "--staLen", type=int, default=3, help="the width of window")
    parser.add_argument("-oL", "--overlap", type=int, default=1, help="the overlap of window")
    parser.add_argument("-sW", "--staWin", type=str, default='hamming', help="window's function")
    parser.add_argument("-izcrt", "--IZCRT", type=float, default=0.1,
                        help="identification zero crossing rate threshold")
    parser.add_argument("-itu", "--ITU", type=float, default=650, help="identification threshold upper")
    parser.add_argument("-alpha", "--alpha", type=float, default=1.3,
                        help="Parameters for automatic adjustment of ITU and IZCRT.")
    parser.add_argument("-noiseT", "--t_backNoise", type=int, default=1e4, help="background noise assessment duration")

    opt = parser.parse_args()
    opt.featuresFold = f'{opt.featuresFold}_sL{opt.staLen}_oL{opt.overlap}'
    print("=" * 44 + " Parameters " + "=" * 44)
    print(opt)

    if opt.first:
        file_list = sorted(os.listdir(opt.streamFold), key=lambda x: int(x.split('-')[-2]))
    else:
        file_list = []
        for file in os.listdir(opt.saveFold):
            if file != 'log':
                file_list.append(f"{file.split('_')[0]}_ch1.mat")
        file_list = list(set(file_list))

    # Compare the initial folder to detect the uncalculated streaming file in the current log file
    if opt.detection:
        with open(os.path.join(opt.saveFold, 'log'), 'r') as f:
            for _ in range(10):
                f.readline()
            calculatedFiles = [i.strip() for i in f.readlines()]

        notCalculated = []
        for i in file_list:
            if i not in calculatedFiles:
                notCalculated.append(i)
        file_list = notCalculated

    each_core = int(math.ceil(len(file_list) / float(opt.processor)))

    if not os.path.exists(opt.saveFold if opt.first else '%s_%d' % (opt.saveFoldNew, opt.ITU)):
        os.mkdir(opt.saveFold if opt.first else '%s_%d' % (opt.saveFoldNew, opt.ITU))

    if not os.path.exists(opt.featuresFold):
        os.mkdir(opt.featuresFold)

    with open(os.path.join(opt.saveFold if opt.first else '%s_%d' % (opt.saveFoldNew, opt.ITU), 'log'), 'a') as f:
        f.write('Parameters config\n')
        f.write('StaLen\t%d\n' % opt.staLen)
        f.write('Overlap\t%d\n' % opt.overlap)
        f.write('StaWin\t%s\n' % opt.staWin)
        f.write('IZCRT\t%f\n' % opt.IZCRT)
        f.write('ITU\t%d\n' % opt.ITU)
        f.write('Alpha\t%f\n' % opt.alpha)
        f.write('BackNoise time\t%d\n\n' % opt.t_backNoise)
        f.write('Calculated Files\n')

    print("=" * 47 + " Start " + "=" * 46)
    start = time.time()

    # Multiprocessing acceleration
    # pool = multiprocessing.Pool(processes=opt.processor)
    # for idx, i in enumerate(range(0, len(file_list), each_core)):
    #     pool.apply_async(cut_stream, (file_list[i:i + each_core], opt.streamFold,
    #                                   opt.saveFold if opt.first else '%s_%d' % (opt.saveFoldNew, opt.ITU), opt,))
    #
    # pool.close()
    # pool.join()

    cut_stream(file_list, opt.streamFold, opt.saveFold if opt.first else '%s_%d' % (opt.saveFoldNew, opt.ITU), opt)

    end = time.time()
    print("=" * 46 + " Report " + "=" * 46)
    print("Calculation Info--Quantity of streaming data: %s" % len(file_list))
    print("Finishing time: {}  |  Time consumption: {:.3f} min".format(time.asctime(time.localtime(time.time())),
                                                                       (end - start) / 60))
