import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sqlite3


def export_time(result_pri, N_pri, t):
    res = [[] * (len(t)-1)]

    for _ in tqdm(range(N_pri)):
        i = result_pri.fetchone()
        if i[-2] is not None and i[-2] >= 6:
            if t[0] <= i[1] < t[1]:
                res[0].append(i)
            if t[1] <= i[1] < t[2]:
                res[1].append(i)
            if t[2] <= i[1] < t[3]:
                res[2].append(i)

    # save features to file
    for k in range(len(t) - 1):
        with open(path_pri[-6] + '-%d-%d.txt' % (t[k], t[k + 1]), 'w') as f:
            f.write('SetID, TRAI, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts\n')
            for i in res[k]:
                f.write('{}, {}, {:.8f}, {}, {:.7f}, {:.7f}, {:.2f}, {:.2f}, {:.7f}, {:.7f}, {}\n'.format(
                    i[0], i[-1], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9]))
