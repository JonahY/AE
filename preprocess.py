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
    def __init__(self, thr_dB):
        self.thr_V = float('%.8f' % (pow(10, thr_dB / 20) / 5000000))
        self.counts = 1
        self.duration = 0
        self.amplitude = 0
        self.rise_time = 0
        self.energy = 0
        self.RMS = 0

    def skip_n_column(self, file, n=3):
        for _ in range(n):
            file.readline()

    def cal_features(self, dataset, time_label, sample_interval, time):
        valid_wave_idx = np.where(abs(dataset) >= self.thr_V)[0]
        start = time_label[valid_wave_idx[0]]
        end = time_label[valid_wave_idx[-1]]
        self.duration = end - start
        max_idx = np.argmax(abs(dataset))
        self.amplitude = abs(dataset[max_idx])
        self.rise_time = time_label[max_idx] - start
        valid_data = dataset[valid_wave_idx[0]:valid_wave_idx[-1]]
        self.energy = np.sum(np.multiply(pow(valid_data, 2), sample_interval))
        self.RMS = math.sqrt(self.energy / time)

        return valid_data

    def cal_counts(self, valid_data):
        N = len(valid_data)
        for idx in range(1, N):
            if valid_data[idx - 1] <= self.thr_V <= valid_data[idx] \
                    or valid_data[idx] <= self.thr_V <= valid_data[idx - 1]:
                self.counts += 1

    def main(self, file_name, data=[]):
        pbar = tqdm(file_name)
        for name in pbar:
            with open(name, "r") as f:
                self.skip_n_column(f)
                sample_interval = float(f.readline()[29:])
                self.skip_n_column(f)
                points_num = int(f.readline()[36:])
                channel_num = int(f.readline().strip()[16:])
                hit_num = int(f.readline()[12:])
                time = float(f.readline()[14:])
                dataset = np.array([float(i.strip("\n")) for i in f.readlines()[1:]])
                time_label = np.linspace(time, time + sample_interval * (points_num - 1), points_num)

                # calculate the duration, amplitude, rise_time, energy and counts
                valid_data = self.cal_features(dataset, time_label, sample_interval, time)
                self.cal_counts(valid_data)

            data.append('{}, {:.1f}, {}, {:.2f}, {:.2f}, {:.1f}, {:.1f}, {:.8f}, {:.2f}, {}\n'.format(hit_num, time * 1000000, channel_num, self.thr_V * 1000000, self.amplitude * 1000000, self.rise_time * 1000000, self.duration * 1000000, self.energy, self.RMS * 1000000, self.counts))
            pbar.set_description("Calculating: %s" % name.split('_')[2])
            # ID, Time(μs), Chan, Thr(μV), Amp(μV), RiseT(μs), Dur(μs), Eny(aJ), RMS(μV), Counts
            # print("-" * 50)
            # print(hit_num, time * 1000000, channel_num, self.thr_V * 1000000, self.amplitude * 1000000, self.rise_time * 1000000, self.duration * 1000000, self.energy, self.RMS * 1000000, self.counts)

        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="C:\\Users\\Yuan\\Desktop\\CM-6M-o18-2020.10.17-1-60-a-waveform", help="Path of data")
    parser.add_argument("--thr_dB", type=int, default=25, help="Detection threshold")
    parser.add_argument("--threads", type=int, default=4, help="Cpu core number")
    opt = parser.parse_args()
    print(opt)

    os.chdir(opt.data_path)
    file_list = os.listdir(opt.data_path)
    # print(file_list)

    each_core = int(math.ceil(len(file_list) / float(opt.threads)))
    result, data = [], []

    # Multithreading acceleration
    pool = multiprocessing.Pool(processes=opt.threads)
    for i in range(0, len(file_list), each_core):
        process = Preprocessing(opt.thr_dB)
        result.append(pool.apply_async(process.main, (file_list[i:i + each_core],)))

    txt_name = opt.data_path.split('\\')[-1] + '.txt'
    f = open(txt_name, "w")
    f.write("ID, Time(μs), Chan, Thr(μV), Amp(μV), RiseT(μs), Dur(μs), Eny(aJ), RMS(μV), Counts\n")
    pbar = tqdm(result)
    for idx, i in enumerate(pbar):
        tmp = i.get()
        data += tmp
        pbar.set_description("Exporting Data: {}/{}".format(idx + 1, opt.threads))
    data = sorted(data, key=lambda s: int(s.split(',')[0]))
    for i in data:
        f.write(i)
    f.close()
    pool.close()
    pool.join()
    # print(data)
    # print(len(data))