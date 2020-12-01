import sqlite3
from tqdm import tqdm


def sqlite_read(path):
    """
    python读取sqlite数据库文件
    """
    mydb = sqlite3.connect(path)  # 链接数据库
    mydb.text_factory = lambda x: str(x, 'gbk', 'ignore')
    cur = mydb.cursor()  # 创建游标cur来执行SQL语句

    # 获取表名
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    Tables = cur.fetchall()  # Tables 为元组列表

    # 获取表结构的所有信息
    if path[-5:] == 'pridb':
        cur.execute("SELECT * FROM {}".format(Tables[3][0]))
        res = cur.fetchall()[-2][1]
    elif path[-5:] == 'tradb':
        cur.execute("SELECT * FROM {}".format(Tables[1][0]))
        res = cur.fetchall()[-3][1]
    return int(res)


def read_data(result_tra, result_pri, path_pri, path_tra):
    data_tra, data_pri, chan_1, chan_2, chan_3, chan_4 = [], [], [], [], [], []
    N_pri = sqlite_read(path_pri)
    N_tra = sqlite_read(path_tra)
    for _ in tqdm(range(N_tra), ncols=80):
        i = result_tra.fetchone()
        data_tra.append(i)
    for _ in tqdm(range(N_pri), ncols=80):
        i = result_pri.fetchone()
        if i[-2] is not None and i[-2] > 5 and i[-1] > 0:
            data_pri.append(i)
            if i[2] == 1:
                chan_1.append(i)
            if i[2] == 2:
                chan_2.append(i)
            elif i[2] == 3:
                chan_3.append(i)
            elif i[2] == 4:
                chan_4.append(i)
    return data_tra, data_pri, chan_1, chan_2, chan_3, chan_4


def material_status(component, status):
    if component == 'pure':
        if status == 'random':
            # 0.508, 0.729, 1.022, 1.174, 1.609
            idx_select_2 = [105, 94, 95, 109, 102]
            TRAI_select_2 = [4117396, 4115821, 4115822, 4117632, 4117393]
            # -0.264, -0.022
            idx_select_1 = [95, 60]
            TRAI_select_1 = [124104, 76892]
    elif component == 'electrolysis':
        if status == 'random':
            # 0.115, 0.275, 0.297, 0.601, 1.024
            idx_select_2 = [50, 148, 51, 252, 10]
            TRAI_select_2 = [3067, 11644, 3079, 28583, 1501]
            # 0.303, 0.409, 0.534, 0.759, 1.026
            idx_select_1 = [13, 75, 79, 72, 71]
            TRAI_select_1 = [2949, 14166, 14815, 14140, 14090]
        if status == 'amp':
            idx_select_2 = [90, 23, 48, 50, 29]
            TRAI_select_2 = [4619, 2229, 2977, 3014, 2345]

            idx_select_1 = [16, 26, 87, 34, 22]
            TRAI_select_1 = [3932, 7412, 16349, 9001, 6300]
        elif status == 'eny':
            idx_select_2 = [79, 229, 117, 285, 59]
            TRAI_select_2 = [4012, 22499, 7445, 34436, 3282]

            idx_select_1 = [160, 141, 57, 37, 70]
            TRAI_select_1 = [26465, 23930, 11974, 9379, 13667]
    return idx_select_1, idx_select_2, TRAI_select_1, TRAI_select_2


def validation(k):
    # Time, Amp, RiseTime, Dur, Eny, Counts, TRAI
    i = data_tra[k]
    sig = np.multiply(array.array('h', bytes(i[-2])), i[-3] * 1000)
    time = np.linspace(i[0], i[0] + pow(i[-5], -1) * (i[-4] - 1), i[-4])

    thr = i[2]
    valid_wave_idx = np.where(abs(sig) >= thr)[0]
    valid_time = time[valid_wave_idx[0]:(valid_wave_idx[-1] + 1)]
    start = time[valid_wave_idx[0]]
    end = time[valid_wave_idx[-1]]
    duration = (end - start) * pow(10, 6)
    max_idx = np.argmax(abs(sig))
    amplitude = max(abs(sig))
    rise_time = (time[max_idx] - start) * pow(10, 6)
    valid_data = sig[valid_wave_idx[0]:(valid_wave_idx[-1] + 1)]
    energy = np.sum(np.multiply(pow(valid_data, 2), pow(10, 6) / i[3]))
    RMS = math.sqrt(energy / duration)
    count, idx = 0, 1
    N = len(valid_data)
    for idx in range(1, N):
        if valid_data[idx - 1] >= thr > valid_data[idx]:
            count += 1
    # while idx < N:
    #     if min(valid_data[idx - 1], valid_data[idx]) <= thr < max((valid_data[idx - 1], valid_data[idx])):
    #         count += 1
    #         idx += 2
    #         continue
    #     idx += 1
    print(i[0], amplitude, rise_time, duration, energy / pow(10, 4), count, i[-1])


def val_TRAI(data_pri, TRAI):
    # Time, Amp, RiseTime, Dur, Eny, Counts, TRAI
    for i in TRAI:
        vallen = data_pri[i - 1]
        print('-' * 80)
        print('{:.8f} {} {} {} {} {:.0f} {:.0f}'.format(vallen[1], vallen[4], vallen[5], vallen[6],
                                                        vallen[-4], vallen[-2], vallen[-1]))
        validation(i - 1)


def save_E_T(Time, Eny, cls_1_KKM, cls_2_KKM):
    df_1 = pd.DataFrame({'time': Time[cls_1_KKM], 'energy': Eny[cls_1_KKM]})
    df_2 = pd.DataFrame({'time': Time[cls_2_KKM], 'energy': Eny[cls_2_KKM]})
    df_1.to_csv('E-T_pure_pop1.csv')
    df_2.to_csv('E-T_pure_pop2.csv')
