# https://github.com/OverLordGoldDragon/ssqueezepy
# pip install ssqueezepy

from ssqueezepy import ssq_cwt


def swt(time, sig, sampleRate, wavelet='morlet', scales='log-piecewise'):
    '''
    :param time:
    :param sig:
    :param sampleRate:
    :param wavelet:
    :param scales:
    :return:
    '''
    Twxo, Wxo, ssq_freqs, *_ = ssq_cwt(sig, wavelet=wavelet, scales=scales, fs=sampleRate, t=time)
    fig = plt.figure(figsize=(5.12, 5.12))
    plt.contourf(time, ssq_freqs * 1000, pow(abs(Twxo), 0.5), cmap='cubehelix_r')
    plt.ylim(min(ssq_freqs * 1000), 1000)
    plt.xlabel(r'Time (μs)')
    plt.ylabel(r'Frequency (kHz)')
