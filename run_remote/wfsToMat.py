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


def convert_stream(files, streamFold, saveFold):
    try:
        pbar = tqdm(files, ncols=100)

        for file in pbar:
            pbar.set_description('File name: %s' % file[43:-4])

            if not os.path.exists(os.path.join(saveFold, f'{file[:-4]}.mat')):
                with open(os.path.join(streamFold, file), 'r') as f:
                    for _ in range(4):
                        f.readline()
                    fs = int(f.readline().strip().split()[-1]) * 1e-3
                    for _ in range(2):
                        f.readline()
                    trigger_time = float(f.readline().strip()[15:])
                    sig = np.array(list(map(lambda x: float(x.strip()) * 1e4, f.readlines()[1:-1])))

                scio.savemat(os.path.join(saveFold, f'{file[:-4]}.mat'), {'Sampling rate': fs, 'Trigger time': trigger_time,
                                                                          'Voltage': sig})

            with open(os.path.join(saveFold, 'log'), 'a') as f:
                f.write(f'{file}\n')

    except Exception as e:
        print('Error: %s' % e)
        print(traceback.format_exc())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-streamF", "--streamFold", type=str, default='/mnt/yuanbincheng/Stream/threshold',
                        help="Absolute path of streaming folder(add 'r' in front)")
    parser.add_argument("-lastF", "--lastFold", type=str,
                        default=r'/home/Yuanbincheng/data/stream/alpha_1.3/waveforms_650_old',
                        help="Absolute path of last calculated folder(add 'r' in front)")
    parser.add_argument("-saveF", "--saveFold", type=str,
                        default=r'/home/Yuanbincheng/data/stream/threshold',
                        help="Absolute path of storage folder(add 'r' in front)")
    parser.add_argument("-f", "--first", type=int, default=1, choices=[0, 1],
                        help="Only the [1] is passed in for the first calculation, and only the streaming file that "
                             "appears in the storage location needs to be calculated later.")
    parser.add_argument("-cpu", "--processor", type=int, default=cpu_count(), help="Number of Threads")
    parser.add_argument("-detect", "--detection", type=int, default=0, choices=[0, 1], help="Whether to detect log file")
    opt = parser.parse_args()
    print("=" * 44 + " Parameters " + "=" * 44)
    print(opt)

    if opt.first:
        file_list = sorted(os.listdir(opt.streamFold), key=lambda x: int(x.split('-')[-2]))
    else:
        file_list = []
        for file in os.listdir(opt.lastFold):
            if file != 'log':
                file_list.append(f"{file.split('_')[0]}_ch1.txt")
        file_list = list(set(file_list))

    # Compare the initial folder to detect the uncalculated streaming file in the current log file
    if opt.detection:
        with open(os.path.join(opt.saveFold, 'log'), 'r') as f:
            calculatedFiles = [i.strip() for i in f.readlines()[1:]]

        notCalculated = []
        for i in file_list:
            if i not in calculatedFiles:
                notCalculated.append(i)
        file_list = notCalculated

    each_core = int(math.ceil(len(file_list) / float(opt.processor)))

    if not os.path.exists(opt.saveFold):
        os.mkdir(opt.saveFold)

    with open(os.path.join(opt.saveFold, 'log'), 'a') as f:
        f.write('Converted Files\n')

    print("=" * 47 + " Start " + "=" * 46)
    start = time.time()

    # Multiprocessing acceleration
    pool = multiprocessing.Pool(processes=opt.processor)
    for idx, i in enumerate(range(0, len(file_list), each_core)):
        pool.apply_async(convert_stream, (file_list[i:i + each_core], opt.streamFold, opt.saveFold, ))

    pool.close()
    pool.join()

    # convert_stream(file_list, opt.streamFold, opt.saveFold)

    end = time.time()
    print("=" * 46 + " Report " + "=" * 46)
    print("Calculation Info--Quantity of streaming data: %s" % len(file_list))
    print("Finishing time: {}  |  Time consumption: {:.3f} min".format(time.asctime(time.localtime(time.time())),
                                                                       (end - start) / 60))
