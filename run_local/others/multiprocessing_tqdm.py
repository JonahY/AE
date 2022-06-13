import time
from tqdm import tqdm
import multiprocessing as mp


def pickle_process(_class, *args):
    return _class.proc_func(*args)


class OP():
    def __init__(self):
        self.length = 64

    def proc_func(self):
        time.sleep(0.1)

    def flow(self):
        # ------------- 配置好进度条 -------------
        pbar = tqdm(total=self.length)
        pbar.set_description(' Flow ')
        update = lambda *args: pbar.update()
        # --------------------------------------
        pool = mp.Pool(4)
        for _ in range(self.length):
            pool.apply_async(pickle_process, args=(self, ), callback=update) # 通过callback来更新进度条
        pool.close()
        pool.join()


if __name__ == '__main__':

    start_time = time.time()
    op = OP()
    op.flow()
    print(time.time() - start_time)