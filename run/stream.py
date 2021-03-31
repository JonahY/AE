def Windows(width, parameter):
    function = np.zeros(width)
    for n in range(width):
        function[n] = (1 - parameter) - parameter * np.cos((2 * 3.14 * n) / (width - 1))
    return function


def framing(audio_1, N, move, hamming):
    framing = np.zeros(1)
    for i in tqdm(range(0, len(audio_1), move)):
        if len(audio_1[i:i + N]) == N:
            tmp = audio_1[i:i + N] * hamming
            framing = np.append(framing, tmp)
        else:
            tmp = audio_1[i:i + N] * hamming[0:len(audio_1[i:i + N])]
            framing = np.append(framing, tmp)
    return framing


def sgn(a):
    if a >= 0:
        return 1
    else:
        return -1


def shortTermEny(audio_1, N, move, fs, hamming):
    short_power = np.zeros(1)
    sample_interval = 1 / fs
    for i in range(0, len(audio_1), move):
        if len(audio_1[i:i + N]) == N:
            tmp = np.sum(np.multiply(pow(np.abs(audio_1[i:i + N]), 2), sample_interval)) * hamming
            short_power = np.append(short_power, tmp)
        else:
            tmp = np.sum(np.multiply(pow(np.abs(audio_1[i:i + N]), 2), sample_interval)) * hamming[
                                                                                           0:len(audio_1[i:i + N])]
            short_power = np.append(short_power, tmp)
    return short_power


# def shortTermEny(audio_1, N, move):
#     short_power = np.zeros(1)
#     for i in range(0, len(audio_1), move):
#         if len(audio_1[i:i + N]) == N:
#             tmp = pow(np.abs(audio_1[i:i + N]), 2) * hamming
#             short_power = np.append(short_power, tmp)
#         else:
#             tmp = pow(np.abs(audio_1[i:i + N]), 2) * hamming[0:len(audio_1[i:i + N])]
#             short_power = np.append(short_power, tmp)
#     return short_power


def zerosCountRate(audio_1, N, move):
    count = np.zeros(1)
    for i in range(0, len(audio_1), move):
        Calculation = 0
        if len(audio_1[i:i + N]) == N:
            for j in range(N):
                tmp = 0.5 * (np.abs(sgn(audio_1[i + j]) - sgn(audio_1[i + j - 1])))
                Calculation += tmp
            count = np.append(count, Calculation)
        else:
            for j in range(len(audio_1[i:i + N])):
                tmp = 0.5 * (np.abs(sgn(audio_1[i + j]) - sgn(audio_1[i + j - 1])))
                Calculation += tmp
            count = np.append(count, Calculation)
    return count


# width_stE = int(tmp[3] * pow(10, -6) * 5)
# width_zcR = int(tmp[3] * pow(10, -6) * 5)
# hamming = Windows(width_stE, 0.46)
#
# stE = shortTermEny(sig, int(width_stE), int(width_stE) - 1, 20, hamming)
# t_stE = np.linspace(time[0], time[-1], stE.shape[0])
# # sig_zcR = framing(sig,int(width_stE),int(width_stE) - 1)
# # zcR = zerosCountRate(sig_zcR, int(width_zcR), int(width_zcR) - 1)
# zcR = zerosCountRate(sig, int(width_zcR), int(width_zcR) - 1)
# t_zcR = np.linspace(time[0], time[-1], zcR.shape[0])
# stE_dev1 = cal_deriv(t_stE, stE)
#
# plt.figure(0)
# plt.plot(time, sig)
# plt.figure(1)
# plt.plot(t_stE, stE)
# plt.figure(2)
# plt.plot(t_zcR, zcR)
# plt.figure(3)
# plt.plot(t_stE, stE_dev1)