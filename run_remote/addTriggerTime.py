import numpy as np
import time
from tqdm import tqdm
import sys
import os
import multiprocessing
from multiprocessing import cpu_count
import math
import argparse


def cut_stream(files, streamFold, saveFold):
    pbar = tqdm(files, ncols=100)
    for file in pbar:
        pbar.set_description('File name: %s' % file[43:-4])

        with open(os.path.join(streamFold, file), 'r') as f:
            for _ in range(7):
                f.readline()
            trigger_time = float(f.readline().strip()[15:])

        for j in os.listdir(saveFold):
            if file[43:-4] in j:
                with open(os.path.join(saveFold, j), "r+") as f:
                    old = f.read()
                    f.seek(0)
                    f.write('Trigger time (s)\n%.8f\n\n' % trigger_time)
                    f.write(old)

                with open(os.path.join(saveFold, 'log'), 'a') as f:
                    f.write('%s\n' % file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-streamF", "--streamFold", type=str, default='/mnt/yuanbincheng/Stream/threshold',
                        help="Absolute path of streaming folder(add 'r' in front)")
    parser.add_argument("-saveF", "--saveFold", type=str, default=r'/home/Yuanbincheng/data/stream/waveforms_650',
                        help="Absolute path of storage folder(add 'r' in front)")
    parser.add_argument("-cpu", "--processor", type=int, default=cpu_count(), help="Number of Threads")
    opt = parser.parse_args()
    print("=" * 44 + " Parameters " + "=" * 44)
    print(opt)

    file_list = []
    for file in os.listdir(opt.saveFold):
        if file != 'log':
            file_list.append('%s_ch1.txt' % file.split('_')[0])
    file_list = list(set(file_list))
    each_core = int(math.ceil(len(file_list) / float(opt.processor)))

    print("=" * 47 + " Start " + "=" * 46)
    start = time.time()

    # Multiprocessing acceleration
    pool = multiprocessing.Pool(processes=opt.processor)
    for idx, i in enumerate(range(0, len(file_list), each_core)):
        pool.apply_async(cut_stream, (file_list[i:i + each_core], opt.streamFold, opt.saveFold))

    pool.close()
    pool.join()

    end = time.time()
    print("=" * 46 + " Report " + "=" * 46)
    print("Calculation Info--Quantity of streaming data: %s" % len(file_list))
    print("Finishing time: {}  |  Time consumption: {:.3f} min".format(time.asctime(time.localtime(time.time())),
                                                                       (end - start) / 60))
