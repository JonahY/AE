import os
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import multiprocessing
import argparse
import time
from multiprocessing import cpu_count
import sys
from scipy.fftpack import fft


# os.getcwd()
np.seterr(invalid='ignore')


class Preprocessing:
    def __init__(self, idx, thr_dB, magnification_dB, data_path, processor):
        self.idx = idx
        self.thr_dB = thr_dB
        self.magnification_dB = magnification_dB
        self.thr_V = pow(10, self.thr_dB / 20) / pow(10, 6)
        self.counts = 0
        self.duration = 0
        self.amplitude = 0
        self.rise_time = 0
        self.energy = 0
        self.RMS = 0
        self.hit_num = 0
        self.time = 0
        self.channel_num = 0
        self.sample_interval = 0
        self.freq_max = 0
        self.magnification = pow(10, self.magnification_dB / 20)
        self.data_path = data_path
        self.processor = processor

    def skip_n_column(self, file, n=3):
        for _ in range(n):
            file.readline()

    def cal_features(self, dataset, time_label, valid_wave_idx):
        start = time_label[valid_wave_idx[0]]
        end = time_label[valid_wave_idx[-1]]
        self.duration = end - start
        max_idx = np.argmax(abs(dataset))
        self.amplitude = abs(dataset[max_idx])
        self.rise_time = time_label[max_idx] - start
        valid_data = dataset[valid_wave_idx[0]:(valid_wave_idx[-1] + 1)]
        self.energy = np.sum(np.multiply(pow(valid_data, 2), self.sample_interval))
        self.RMS = math.sqrt(self.energy / self.duration)
        return valid_data

    def cal_counts(self, valid_data):
        self.counts = 0
        N = len(valid_data)
        for idx in range(1, N):
            if valid_data[idx - 1] <= self.thr_V <= valid_data[idx]:
                self.counts += 1

    def cal_freq(self, valid_data, valid_wave_idx):
        Fs = 1 / self.sample_interval
        N = valid_wave_idx[-1] - valid_wave_idx[0] + 1
        frq = (np.arange(N) / N) * Fs
        fft_y = fft(valid_data)
        abs_y = np.abs(fft_y) / N
        half_frq = frq[range(int(N / 2))]
        abs_y_half = abs_y[range(int(N / 2))]
        abs_y_half[0] = 0
        self.freq_max = half_frq[np.argmax(abs_y_half)]

    def save_features(self, result):
        valid_data = []
        txt_name = self.data_path.split('/')[-1] + '.txt'
        f = open(txt_name, "w")
        f.write("ID, Time(s), Chan, Thr(μV), Thr(dB), Amp(μV), Amp(dB), "
                "RiseT(s), Dur(s), Eny(aJ), RMS(μV), Counts, Frequency(Hz)\n")
        pbar = tqdm(result, ncols=100)
        for idx, i in enumerate(pbar):
            tmp = i.get()
            valid_data += tmp
            pbar.set_description("Exporting Data: {}/{}".format(idx + 1, self.processor))
        valid_data = sorted(valid_data, key=lambda s: float(s.split(',')[0]))
        for i in valid_data:
            f.write(i)
        f.close()
        # print(valid_data)
        return len(valid_data)

    def main(self, file_name, data=[]):
        pbar = tqdm(file_name, ncols=100)
        for name in pbar:
            with open(name, "r") as f:
                self.skip_n_column(f)
                self.sample_interval = float(f.readline()[29:])
                self.skip_n_column(f)
                points_num = int(f.readline()[36:])
                self.channel_num = int(f.readline().strip()[16:])
                self.hit_num = int(f.readline()[12:])
                self.time = float(f.readline()[14:])
                dataset = np.array([float(i.strip("\n")) for i in f.readlines()[1:]]) / self.magnification
                time_label = np.linspace(self.time, self.time + self.sample_interval * (points_num - 1), points_num)

                # calculate the duration, amplitude, rise_time, energy and counts
                valid_wave_idx = np.where(abs(dataset) >= self.thr_V)[0]
                # print(dataset[0], dataset[-1], len(dataset))
                # print(valid_wave_idx, valid_wave_idx.shape)

                if valid_wave_idx.shape[0] > 1:
                    valid_data = self.cal_features(dataset, time_label, valid_wave_idx)
                    self.cal_counts(valid_data)
                    if self.counts >= 2:
                        self.cal_freq(valid_data, valid_wave_idx)
                        data.append('{}, {:.7f}, {}, {:.8f}, {:.1f}, {:.8f}, {:.1f}, {:.7f}, {:.7f}, {:.8f}, {:.8f}, '
                                    '{}, {}\n'.format(self.hit_num, self.time, self.channel_num,
                                                      self.thr_V * pow(10, 6), self.thr_dB, self.amplitude * pow(10, 6),
                                                      20 * np.log10(self.amplitude * pow(10, 6)), self.rise_time,
                                                      self.duration, self.energy * pow(10, 14), self.RMS * pow(10, 6),
                                                      self.counts, self.freq_max))
            pbar.set_description("Process: %s | Calculating: %s" % (self.idx, name.split('_')[2]))
            # ID, Time(s), Chan, Thr(μV)P, Thr(dB), Amp(μV), Amp(dB), RiseT(s), Dur(s), Eny(aJ), RMS(μV), Counts
            # print("-" * 50)
            # print(self.hit_num, self.time * pow(10, 6), self.channel_num, self.thr_V * pow(10, 6),
            #     self.amplitude * pow(10, 6), self.rise_time * pow(10, 6), self.duration * pow(10, 6),
            #     self.energy * pow(10, 14), self.RMS * pow(10, 6), self.counts)

        return data

    def read_pac_data(self, file_name, tra_1=[], tra_2=[], tra_3=[], tra_4=[]):
        pbar = tqdm(file_name, ncols=100)
        for name in pbar:
            with open(name, "r") as f:
                self.skip_n_column(f)
                self.sample_interval = float(f.readline()[29:])
                self.skip_n_column(f)
                points_num = int(f.readline()[36:])
                self.channel_num = int(f.readline().strip()[16:])
                self.hit_num = int(f.readline()[12:])
                self.time = float(f.readline()[14:])
                dataset = np.array([float(i.strip("\n")) for i in f.readlines()[1:]]) / self.magnification
            if self.channel_num == 1:
                tra_1.append([self.time, self.channel_num, self.sample_interval, points_num, dataset, self.hit_num])
            elif self.channel_num == 2:
                tra_2.append([self.time, self.channel_num, self.sample_interval, points_num, dataset, self.hit_num])
            elif self.channel_num == 3:
                tra_3.append([self.time, self.channel_num, self.sample_interval, points_num, dataset, self.hit_num])
            elif self.channel_num == 4:
                tra_4.append([self.time, self.channel_num, self.sample_interval, points_num, dataset, self.hit_num])
        return tra_1, tra_2, tra_3, tra_4

    def read_pac_features(self, dir_features, min_cnts=2):
        with open(dir_features, 'r') as f:
            res = np.array([i.strip("\n").strip(',') for i in f.readlines()[1:]])
        pri, chan_1, chan_2, chan_3, chan_4 = [], [], [], [], []
        for i in tqdm(res):
            tmp = []
            ls = i.strip("\n").split(', ')
            if int(ls[-1]) > min_cnts:
                for j in ls:
                    tmp.append(float(j))
                pri.append(tmp)
                if int(ls[2]) == 1:
                    chan_1.append(tmp)
                elif int(ls[2]) == 2:
                    chan_2.append(tmp)
                elif int(ls[2]) == 3:
                    chan_3.append(tmp)
                elif int(ls[2]) == 4:
                    chan_4.append(tmp)
        data_pri = np.array(pri)
        chan_1 = np.array(chan_1)
        chan_2 = np.array(chan_2)
        chan_3 = np.array(chan_3)
        chan_4 = np.array(chan_4)
        return data_pri, chan_1, chan_2, chan_3, chan_4


def convert_pac_data(file_list, data_path, processor, threshold_dB, magnification_dB):
    # check existing file
    tar = data_path.split('/')[-1] + '.txt'
    if tar in file_list:
        print("=" * 46 + " Warning " + "=" * 45)
        while True:
            ans = input("The exported data file has been detected. Do you want to overwrite it: (Enter 'yes' or 'no') ")
            if ans.strip() == 'yes':
                os.remove(tar)
                break
            elif ans.strip() == 'no':
                sys.exit(0)
            print("Please enter 'yes' or 'no' to continue!")

    file_list = os.listdir(data_path)
    each_core = int(math.ceil(len(file_list) / float(processor)))
    result = []

    print("=" * 47 + " Start " + "=" * 46)
    start = time.time()

    # Multiprocessing acceleration
    pool = multiprocessing.Pool(processes=processor)
    for idx, i in enumerate(range(0, len(file_list), each_core)):
        process = Preprocessing(idx, threshold_dB, magnification_dB, data_path, processor)
        result.append(pool.apply_async(process.main, (file_list[i:i + each_core],)))

    N = process.save_features(result)

    pool.close()
    pool.join()

    end = time.time()
    print("=" * 46 + " Report " + "=" * 46)
    print("Quantity of valid data: %s" % N)
    print("Finishing time: {}  |  Time consumption: {:.3f} min".format(time.asctime(time.localtime(time.time())),
                                                                     (end - start) / 60))


def multiprocess_read_pac_data(file_list, data_path, magnification_dB, threshold_dB, processor):
    each_core = int(math.ceil(len(file_list) / float(processor)))
    result, tra_1, tra_2, tra_3, tra_4 = [], [], [], [], []
    data_tra = []
    # Multiprocessing acceleration
    pool = multiprocessing.Pool(processes=processor)
    for idx, i in enumerate(range(0, len(file_list), each_core)):
        process = Preprocessing(idx, threshold_dB, magnification_dB, data_path, processor)
        result.append(pool.apply_async(process.read_pac_data, (file_list[i:i + each_core],)))

    pbar = tqdm(result, ncols=100)
    for idx, i in enumerate(pbar):
        tmp_1, tmp_2, tmp_3, tmp_4 = i.get()
        tra_1.append(tmp_1)
        tra_2.append(tmp_2)
        tra_3.append(tmp_3)
        tra_4.append(tmp_4)
        pbar.set_description("Exporting Data: {}/{}".format(idx + 1, processor))

    pool.close()
    pool.join()

    for tra in [tra_1, tra_2, tra_3, tra_4]:
        tra = [j for i in tra for j in i]
        try:
            data_tra.append(sorted(tra, key=lambda x: x[-1]))
        except IndexError:
            data_tra.append([])
            print('There is no data in this channel!')
    return data_tra[0], data_tra[1], data_tra[2], data_tra[3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "--data_path", type=str,
                        default=r"E:\data\CM-6M-o18-2020.10.17-1-60",
                        help="Absolute path of data(add 'r' in front)")
    parser.add_argument("-thr", "--threshold_dB", type=int, default=25, help="Detection threshold")
    parser.add_argument("-mag", "--magnification_dB", type=int, default=60, help="Magnification /dB")
    parser.add_argument("-cpu", "--processor", type=int, default=cpu_count(), help="Number of Threads")
    opt = parser.parse_args()
    print("=" * 44 + " Parameters " + "=" * 44)
    print(opt)

    opt.data_path = opt.data_path.replace('\\', '/')
    os.chdir(opt.data_path)
    file_list = os.listdir(opt.data_path)[:100]
    # print(file_list)

    convert_pac_data(file_list, opt.data_path, opt.processor, opt.threshold_dB, opt.magnification_dB)

    # data_tra_1, data_tra_2, data_tra_3, data_tra_4 = multiprocess_read_pac_data(file_list, opt.data_path, opt.magnification_dB, opt.threshold_dB, opt.processor)