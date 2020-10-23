import os
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import multiprocessing
import argparse
import time


# os.getcwd()


class Preprocessing:
    def __init__(self, thr_dB, thr_noise_ratio, magnification_dB):
        self.thr_dB = thr_dB
        self.thr_noise_ratio = thr_noise_ratio
        self.magnification_dB = magnification_dB
        self.thr_V = pow(10, self.thr_dB / 20) / (self.thr_noise_ratio * pow(10, 6))
        self.counts = 1
        self.duration = 0
        self.amplitude = 0
        self.rise_time = 0
        self.energy = 0
        self.RMS = 0
        self.hit_num = 0
        self.time = 0
        self.channel_num = 0
        self.sample_interval = 0
        self.magnification = pow(10, self.magnification_dB / 20)

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
        valid_data = dataset[valid_wave_idx[0]:valid_wave_idx[-1]]
        self.energy = np.sum(np.multiply(pow(valid_data, 2), self.sample_interval))
        self.RMS = math.sqrt(self.energy / self.time)

        return valid_data

    def cal_counts(self, valid_data):
        N = len(valid_data)
        for idx in range(1, N):
            if valid_data[idx - 1] <= self.thr_V <= valid_data[idx]:
                self.counts += 1

    def main(self, file_name, data=[]):
        pbar = tqdm(file_name, ncols=80)
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
                # print(valid_wave_idx)

                if valid_wave_idx.shape[0]:
                    valid_data = self.cal_features(dataset, time_label, valid_wave_idx)
                    self.cal_counts(valid_data)
                    if self.counts >= 2:
                        data.append(
                            '{}, {:.7f}, {}, {:.8f}, {:.1f}, {:.8f}, {:.1f}, {:.7f}, {:.7f}, {:.8f}, {:.8f}, {}\n'.format(
                                self.hit_num, self.time, self.channel_num, self.thr_V * pow(10, 6), self.thr_dB,
                                                                           self.amplitude * pow(10, 6),
                                                                           20 * np.log10(
                                                                               self.thr_noise_ratio * self.amplitude * pow(
                                                                                   10, 6)), self.rise_time,
                                self.duration, self.energy * pow(10, 14), self.RMS * pow(10, 6), self.counts))
            pbar.set_description("Calculating: %s" % name.split('_')[2])
            # ID, Time(s), Chan, Thr(μV), Thr(dB), Amp(μV), Amp(dB), RiseT(s), Dur(s), Eny(aJ), RMS(μV), Counts
            # print("-" * 50)
            # print(self.hit_num, self.time * pow(10, 6), self.channel_num, self.thr_V * pow(10, 6),
            #     self.amplitude * pow(10, 6), self.rise_time * pow(10, 6), self.duration * pow(10, 6),
            #     self.energy * pow(10, 14), self.RMS * pow(10, 6), self.counts)

        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default=r'C:/Users/Yuan/Desktop/CM-6M-o18-2020.10.17-1-60-a-waveform',
                        help="Absolute path of data(add 'r' in front)")
    parser.add_argument("--thr_dB", type=int, default=25, help="Detection threshold")
    parser.add_argument("--thr_noise_ratio", type=int, default=5, help="Threshold to noise ratio")
    parser.add_argument("--magnification_dB", type=int, default=60, help="Magnification /dB")
    parser.add_argument("--threads", type=int, default=4, help="Cpu core number")
    opt = parser.parse_args()
    print(opt)

    opt.data_path = opt.data_path.replace('\\', '/')
    os.chdir(opt.data_path)
    file_list = os.listdir(opt.data_path)
    # print(file_list)

    each_core = int(math.ceil(len(file_list) / float(opt.threads)))
    result, data = [], []

    # Multithreading acceleration
    pool = multiprocessing.Pool(processes=opt.threads)
    for i in range(0, len(file_list), each_core):
        process = Preprocessing(opt.thr_dB, opt.thr_noise_ratio, opt.magnification_dB)
        result.append(pool.apply_async(process.main, (file_list[i:i + each_core],)))

    txt_name = opt.data_path.split('/')[-1] + '.txt'
    f = open(txt_name, "w")
    f.write("ID, Time(s), Chan, Thr(μV), Thr(dB), Amp(μV), Amp(dB), RiseT(s), Dur(s), Eny(aJ), RMS(μV), Counts\n")
    pbar = tqdm(result, ncols=80)
    for idx, i in enumerate(pbar):
        tmp = i.get()
        data += tmp
        pbar.set_description("Exporting Data: {}/{}".format(idx + 1, opt.threads))
    data = sorted(data, key=lambda s: float(s.split(',')[0]))
    for i in data:
        f.write(i)
    f.close()
    pool.close()
    pool.join()
    # print(data)
    print(len(data))
