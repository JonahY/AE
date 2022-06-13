import sqlite3
from tqdm import tqdm
import numpy as np
import os
import json
import sys


class Reload:
    def __init__(self, path_pri: str, path_tra: str, fold: str):
        """
        数据载入类
        :param path_pri: .pridb数据库文件全称
        :param path_tra: .tradb数据库文件全称
        :param fold: 数据库文件名
        """
        self.path_pri = path_pri
        self.path_tra = path_tra
        self.fold = fold

    def sqlite_read(self, path, mode='vallen'):
        """
        读取sqlite数据库文件
        :param path: 文件绝对路径
        :param mode: 读取模式，可选参数：['vallen', 'stream']
                    vallen: 适用于vallen数据库
                    stream: 适用于波形流数据库
        :return:
        """
        mydb = sqlite3.connect(path)
        mydb.text_factory = lambda x: str(x, 'gbk', 'ignore')
        cur = mydb.cursor()

        # 获取表名
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        Tables = cur.fetchall()

        # 获取表结构的所有信息
        if mode == 'vallen':
            if path[-5:] == 'pridb':
                cur.execute("SELECT * FROM {}".format(Tables[3][0]))
                res = cur.fetchall()[-2][1]
            elif path[-5:] == 'tradb':
                cur.execute("SELECT * FROM {}".format(Tables[1][0]))
                res = cur.fetchall()[-3][1]
        elif mode == 'stream':
            cur.execute("SELECT * FROM {}".format(Tables[1][0]))
            res = cur.fetchall()[-1][1]
        return int(res)

    def read_vallen_data(self, lower=2, t_str=0, t_cut=float('inf'), mode='all'):
        """
        vallen数据库载入主函数
        :param lower: 载入时counts的阈值
        :param t_str: 载入特定时间范围的开始时刻
        :param t_cut: 载入特定时间范围的终止时刻
        :param mode: 载入模式，可选参数：['all', 'pri only', 'tra only']
                    all: 同时载入波形文件.tradb与特征文件.pridb
                    pri only: 仅载入特征文件
                    tra only: 仅载入波形文件
        :return:
        """
        data_tra, data_pri, chan_1, chan_2, chan_3, chan_4 = [], [], [], [], [], []
        if mode == 'all' or mode == 'tra only':
            conn_tra = sqlite3.connect(self.path_tra)
            result_tra = conn_tra.execute(
                "Select Time, Chan, Thr, SampleRate, Samples, TR_mV, Data, TRAI FROM view_tr_data")
            N_tra = self.sqlite_read(self.path_tra)
            for _ in tqdm(range(N_tra), ncols=80):
                i = result_tra.fetchone()
                if t_str <= i[0] <= t_cut:
                    data_tra.append(i)
                elif i[0] > t_cut:
                    break
        if mode == 'all' or mode == 'pri only':
            conn_pri = sqlite3.connect(self.path_pri)
            result_pri = conn_pri.execute(
                "Select SetID, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts, TRAI FROM view_ae_data")
            N_pri = self.sqlite_read(self.path_pri)
            for _ in tqdm(range(N_pri), ncols=80):
                i = result_pri.fetchone()
                if t_str <= i[1] <= t_cut:
                    if i[-2] is not None and i[-2] > lower and i[-1] > 0:
                        data_pri.append(i)
                        if i[2] == 1:
                            chan_1.append(i)
                        elif i[2] == 2:
                            chan_2.append(i)
                        elif i[2] == 3:
                            chan_3.append(i)
                        elif i[2] == 4:
                            chan_4.append(i)
                elif i[1] > t_cut:
                    break
        data_tra = sorted(data_tra, key=lambda x: x[-1])  # 按AE事件编号（TRAI）排序便于后续索引
        data_pri = np.array(data_pri)
        chan_1 = np.array(chan_1)
        chan_2 = np.array(chan_2)
        chan_3 = np.array(chan_3)
        chan_4 = np.array(chan_4)
        return data_tra, data_pri, chan_1, chan_2, chan_3, chan_4


if __name__ == "__main__":
    """
    AE数据的说明文件需放在此脚本同目录下，命名为“metarialsInfo.json”
    格式：
        {
          "316L": {
            "AM-Cu-20220328-test1-tension-0.05mm-min": {
              "t_str": 0,
              "t_cut": "inf"
            }
        }
    说明：
        1. 先按不同材料类别进行划分，key为材料类别，value字典用于存储对应材料的AE数据信息
        2. 同一材料中将字典的key命名为数据库名字，value字典包含但不限于"t_str"和"t_cut"两个参数。
            "t_str"参数只能为整数，用于特定时间范围筛选的起始时刻。
            "t_cut"参数只能为整数和"inf"，用于特定时间范围筛选的终止时刻。
    """
    with open('./metarialsInfo.json', 'r', encoding='utf-8') as f:
        js = json.load(f)

    path = r'F:\VALLEN\316L'  # 数据库文件夹上一级的绝对路径
    fold = "AM-Cu-20220328-test1-tension-0.05mm-min"  # 数据库名
    info = js['Cu'][fold]  # 需与说明文件中的材料类别相对应
    path_pri = fold + '.pridb'
    path_tra = fold + '.tradb'
    features_path = fold + '.txt'
    os.chdir('/'.join([path, fold]))

    # 说明文件入参检测
    try:
        for param in ['t_str', 't_cut']:
            if param in info.keys():
                if (type(info[param]) != int) and (info[param] != 'inf'):
                    raise Exception(
                        f"Check the type of the '{param}' input parameter in the database specification file.")
            else:
                raise Exception(f"No '{param}' parameter in the database specification file.")
    except Exception as e:
        print(e)
        sys.exit(0)

    reload = Reload(path_pri, path_tra, fold)
    data_tra, data_pri, chan_1, chan_2, chan_3, chan_4 = reload.read_vallen_data(lower=2, mode='all',
                                                                                 t_str=0, t_cut=float('inf'))
    print('Channel 1: {} | Channel 2: {} | Channel 3: {} | Channel 4: {}'.format(chan_1.shape[0], chan_2.shape[0],
                                                                                 chan_3.shape[0], chan_4.shape[0]))
