# ================================================ Depiction of method =================================================
t_initial, sig_initial = waveform.cal_wave(data_tra[18481 - 1], False)


staLen, overlap, staWin, IZCRT, ITU, alpha, t_backNoise = 2, 0, 'hamming', 0.1, 650, 0.4, 0
t_str, t_end = 0, 1e7
t = t_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1] - t_initial[int(t_str // t_initial[1])]
sig = sig_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1]

width = int(20 * staLen)
stride = int(width) - overlap
t_stE, stE, zcR = shortTermEny_zerosCrossingRate(sig, width, stride, 20, staWin)
stE_dev = cal_deriv(t_stE, stE)
# start, end = find_wave(stE, stE_dev, zcR, t_stE, IZCRT=IZCRT, ITU=ITU, alpha=alpha, t_backNoise=t_backNoise)
start, end, startTmp, endTmp, ITUTmp, IZCRTTmp, ITLTmp = find_wave_multiOutput(stE, stE_dev, zcR, t_stE, IZCRT=IZCRT,
                                                                               ITU=ITU, alpha=alpha,
                                                                               t_backNoise=t_backNoise)

x = [t, t_stE, t_stE, t_stE]
y = [sig, stE, stE_dev, zcR]
color = ['black', 'green', 'gray', 'purple']
ylabel = ['Amplitude / μV', r'STEnergy / μV$^2 \cdot$μs', r'S$\dot{T}$E / μV$^2$', r'ST$\widehat{Z}$CR / %']
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
for idx, ax in enumerate(axes):
    ax.plot(x[idx], y[idx], lw=1, color=color[idx])
    if idx == 0:
        for s, e in tqdm(zip(start, end)):
            ax.plot(t[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2],
                    sig[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2], lw=1, color='red')
            print(t[int(t_stE[s] // t[1]) + 1], t[int(t_stE[e] // t[1]) + 1],
                  t[int(t_stE[e] // t[1]) + 1] - t[int(t_stE[s] // t[1]) + 1])
            ax.axvspan(t[int(t_stE[s] // t[1]) + 1], t[int(t_stE[e] // t[1]) + 1], facecolor=[84 / 255, 1, 159 / 255],
                       alpha=0.5)
            ax.axvline(x=t[int(t_stE[s] // t[1]) + 1], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
            ax.axvline(x=t[int(t_stE[e] // t[1]) + 1], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
        ax.set_xticks(np.array(range(0, 251, 25)))
    elif idx == 1:
        ax.set_yticks(np.array(range(-1000, 10000, 2500)))
        ax.fill_between(np.linspace(0, 5, 10), -1000, [500] * 10, facecolor='orange', alpha=0.2)
        ax.fill_between(x[idx], -1000, y[idx], facecolor=[95 / 255, 158 / 255, 160 / 255], alpha=0.2)
        ax.plot(np.linspace(0, x[idx][endTmp[0]], 100), [ITUTmp[0]] * 100, color='orange', linestyle='dashed', lw=1)
        for i in range(len(start)):
            ax.axvspan(x[idx][startTmp[i]], x[idx][endTmp[i]], facecolor=[95 / 255, 158 / 255, 160 / 255], alpha=0.2)
            ax.fill_between(x[idx][startTmp[i]:endTmp[i] + 1], -1000, y[idx][startTmp[i]:endTmp[i] + 1],
                            facecolor=[28 / 255, 28 / 255, 28 / 255], alpha=0.5)
            ax.axvline(x=x[idx][startTmp[i]], color='k', linestyle='dashdot', lw=1)
            ax.axvline(x=x[idx][endTmp[i]], color='orange', linestyle='dashed', lw=1)
            ax.axvline(x=x[idx][end[i]], color=[70 / 255, 130 / 255, 180 / 255], linestyle='solid', lw=1)
            if i:
                ax.plot(np.linspace(x[idx][end[i-1]], x[idx][endTmp[i]], 100), [ITUTmp[i]] * 100, color='orange',
                        linestyle='dashed', lw=1)
                ax.fill_between(np.linspace(x[idx][end[i-1]], x[idx][start[i]], 100), -1000, [500] * 100,
                                facecolor=[0, 1, 1],
                                alpha=0.5)
    elif idx == 2:
        ax.set_yticks(np.array(range(-2000, 2500, 1000)))
        for i in range(len(start)):
            ax.axvspan(x[idx][start[i]], x[idx][startTmp[i]], facecolor=[84 / 255, 1, 159 / 255], alpha=0.5)
            ax.axvline(x=x[idx][startTmp[i]], color='k', linestyle='dashdot', lw=1)
    else:
        ax.set_yticks(np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]))
        ax.axvspan(0, 5, facecolor='orange', alpha=0.2)
        for i in range(len(start)):
            ax.axvspan(x[idx][endTmp[i]], x[idx][end[i]], facecolor=[84 / 255, 1, 159 / 255], alpha=0.5)
            ax.axvline(x=x[idx][endTmp[i]], color='orange', linestyle='dashed', lw=1)
            ax.axvline(x=x[idx][end[i]], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
            ax.plot(np.linspace(x[idx][endTmp[i]], x[idx][end[i]], 100), [IZCRTTmp[i]] * 100,
                    color=[70 / 255, 130 / 255, 180 / 255],
                    linestyle='dashed', lw=1)
            if i:
                ax.axvspan(x[idx][end[i-1]], x[idx][start[i]], facecolor=[0, 1, 1], alpha=0.5)

    ax.grid(linewidth=0.3)
    plot_norm(ax, 'Time / μs' if idx == 3 else '', ylabel[idx], legend=False, labelWeight='normal')

plt.subplots_adjust(wspace=0, hspace=0)


# ===================================================== Same ITU =======================================================
t_initial, sig_initial = waveform.cal_wave(data_tra[167541 - 1], False)

# ----------------------------------------------------- Alpha 1.2 ------------------------------------------------------
staLen, overlap, staWin, IZCRT, ITU, alpha, t_backNoise = 2, 1, 'hamming', 0.3, 650, 0.5, 50
t_str, t_end = 50, 1e7
t = t_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1] - t_initial[int(t_str // t_initial[1])]
sig = sig_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1]

width = int(20 * staLen)
stride = int(width) - overlap
t_stE, stE, zcR = shortTermEny_zerosCrossingRate(sig, width, stride, 20, staWin)
stE_dev = cal_deriv(t_stE, stE)
# start, end = find_wave(stE, stE_dev, zcR, t_stE, IZCRT=IZCRT, ITU=ITU, alpha=alpha, t_backNoise=t_backNoise)
start, end, startTmp, endTmp, ITUTmp, IZCRTTmp, ITLTmp = find_wave_multiOutput(stE, stE_dev, zcR, t_stE, IZCRT=IZCRT,
                                                                               ITU=ITU, alpha=alpha,
                                                                               t_backNoise=t_backNoise)

x = [t, t_stE, t_stE, t_stE]
y = [sig, stE, stE_dev, zcR]
color = ['black', 'green', 'gray', 'purple']
ylabel = ['Amplitude / μV', r'STEnergy / μV$^2 \cdot$μs', r'S$\dot{T}$E / μV$^2$', r'ST$\widehat{Z}$CR / %']
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(5, 8))
for idx, ax in enumerate(axes):
    ax.plot(x[idx], y[idx], lw=1, color=color[idx])
    if idx == 0:
        for s, e in tqdm(zip(start, end)):
            ax.plot(t[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2],
                    sig[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2], lw=1, color='red')
            print(t[int(t_stE[s] // t[1]) + 1], t[int(t_stE[e] // t[1]) + 1],
                  t[int(t_stE[e] // t[1]) + 1] - t[int(t_stE[s] // t[1]) + 1])
            ax.axvspan(t[int(t_stE[s] // t[1]) + 1], t[int(t_stE[e] // t[1]) + 1], facecolor=[84 / 255, 1, 159 / 255],
                       alpha=0.5)
            ax.axvline(x=t[int(t_stE[s] // t[1]) + 1], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
            ax.axvline(x=t[int(t_stE[e] // t[1]) + 1], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
        # ax.set_xticks(np.array(range(0, 251, 25)))
    elif idx == 1:
        # ax.set_yticks(np.array(range(-1000, 10000, 2500)))
        ax.fill_between(np.linspace(0, t_backNoise, 10), -1000, [500] * 10, facecolor='orange', alpha=0.2)
        ax.fill_between(x[idx], -1000, y[idx], facecolor=[95 / 255, 158 / 255, 160 / 255], alpha=0.2)
        ax.plot(np.linspace(t_backNoise, x[idx][endTmp[0]], 100), [ITUTmp[0]] * 100, color='orange', linestyle='dashed', lw=1)
        for i in range(len(start)):
            ax.axvspan(x[idx][startTmp[i]], x[idx][endTmp[i]], facecolor=[95 / 255, 158 / 255, 160 / 255], alpha=0.2)
            ax.fill_between(x[idx][startTmp[i]:endTmp[i] + 1], -1000, y[idx][startTmp[i]:endTmp[i] + 1],
                            facecolor=[28 / 255, 28 / 255, 28 / 255], alpha=0.5)
            ax.axvline(x=x[idx][startTmp[i]], color='k', linestyle='dashdot', lw=1)
            ax.axvline(x=x[idx][endTmp[i]], color='orange', linestyle='dashed', lw=1)
            ax.axvline(x=x[idx][end[i]], color=[70 / 255, 130 / 255, 180 / 255], linestyle='solid', lw=1)
            if i:
                ax.plot(np.linspace(x[idx][end[i-1]], x[idx][endTmp[i]], 100), [ITUTmp[i]] * 100, color='orange',
                        linestyle='dashed', lw=1)
                ax.fill_between(np.linspace(x[idx][end[i-1]], x[idx][start[i]], 100), -1000, [500] * 100,
                                facecolor=[0, 1, 1],
                                alpha=0.5)
    elif idx == 2:
        # ax.set_yticks(np.array(range(-2000, 2500, 1000)))
        for i in range(len(start)):
            ax.axvspan(x[idx][start[i]], x[idx][startTmp[i]], facecolor=[84 / 255, 1, 159 / 255], alpha=0.5)
            ax.axvline(x=x[idx][startTmp[i]], color='k', linestyle='dashdot', lw=1)
    else:
        # ax.set_yticks(np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]))
        ax.axvspan(0, t_backNoise, facecolor='orange', alpha=0.2)
        for i in range(len(start)):
            ax.axvspan(x[idx][endTmp[i]], x[idx][end[i]], facecolor=[84 / 255, 1, 159 / 255], alpha=0.5)
            ax.axvline(x=x[idx][endTmp[i]], color='orange', linestyle='dashed', lw=1)
            ax.axvline(x=x[idx][end[i]], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
            ax.plot(np.linspace(x[idx][endTmp[i]], x[idx][end[i]], 100), [IZCRTTmp[i]] * 100,
                    color=[70 / 255, 130 / 255, 180 / 255],
                    linestyle='dashed', lw=1)
            if i:
                ax.axvspan(x[idx][end[i-1]], x[idx][start[i]], facecolor=[0, 1, 1], alpha=0.5)

    ax.grid(linewidth=0.3)
    plot_norm(ax, 'Time / μs' if idx == 3 else '', ylabel[idx], legend=False, labelWeight='normal')

plt.subplots_adjust(wspace=0, hspace=0)

# ----------------------------------------------------- Alpha 1.3 -----------------------------------------------------
staLen, overlap, staWin, IZCRT, ITU, alpha, t_backNoise = 2, 1, 'hamming', 0.3, 650, 1.0, 50
t_str, t_end = 50, 1e7
t = t_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1] - t_initial[int(t_str // t_initial[1])]
sig = sig_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1]

width = int(20 * staLen)
stride = int(width) - overlap
t_stE, stE, zcR = shortTermEny_zerosCrossingRate(sig, width, stride, 20, staWin)
stE_dev = cal_deriv(t_stE, stE)
# start, end = find_wave(stE, stE_dev, zcR, t_stE, IZCRT=IZCRT, ITU=ITU, alpha=alpha, t_backNoise=t_backNoise)
start, end, startTmp, endTmp, ITUTmp, IZCRTTmp, ITLTmp = find_wave_multiOutput(stE, stE_dev, zcR, t_stE, IZCRT=IZCRT,
                                                                               ITU=ITU, alpha=alpha,
                                                                               t_backNoise=t_backNoise)

x = [t, t_stE, t_stE, t_stE]
y = [sig, stE, stE_dev, zcR]
color = ['black', 'green', 'gray', 'purple']
ylabel = ['Amplitude / μV', r'STEnergy / μV$^2 \cdot$μs', r'S$\dot{T}$E / μV$^2$', r'ST$\widehat{Z}$CR / %']
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(5, 8))
for idx, ax in enumerate(axes):
    ax.plot(x[idx], y[idx], lw=1, color=color[idx])
    if idx == 0:
        for s, e in tqdm(zip(start, end)):
            ax.plot(t[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2],
                    sig[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2], lw=1, color='red')
            print(t[int(t_stE[s] // t[1]) + 1], t[int(t_stE[e] // t[1]) + 1],
                  t[int(t_stE[e] // t[1]) + 1] - t[int(t_stE[s] // t[1]) + 1])
            ax.axvspan(t[int(t_stE[s] // t[1]) + 1], t[int(t_stE[e] // t[1]) + 1], facecolor=[84 / 255, 1, 159 / 255],
                       alpha=0.5)
            ax.axvline(x=t[int(t_stE[s] // t[1]) + 1], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
            ax.axvline(x=t[int(t_stE[e] // t[1]) + 1], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
        # ax.set_xticks(np.array(range(0, 251, 25)))
    elif idx == 1:
        # ax.set_yticks(np.array(range(-1000, 10000, 2500)))
        ax.fill_between(np.linspace(0, t_backNoise, 10), -1000, [500] * 10, facecolor='orange', alpha=0.2)
        ax.fill_between(x[idx], -1000, y[idx], facecolor=[95 / 255, 158 / 255, 160 / 255], alpha=0.2)
        ax.plot(np.linspace(t_backNoise, x[idx][endTmp[0]], 100), [ITUTmp[0]] * 100, color='orange', linestyle='dashed', lw=1)
        for i in range(len(start)):
            ax.axvspan(x[idx][startTmp[i]], x[idx][endTmp[i]], facecolor=[95 / 255, 158 / 255, 160 / 255], alpha=0.2)
            ax.fill_between(x[idx][startTmp[i]:endTmp[i] + 1], -1000, y[idx][startTmp[i]:endTmp[i] + 1],
                            facecolor=[28 / 255, 28 / 255, 28 / 255], alpha=0.5)
            ax.axvline(x=x[idx][startTmp[i]], color='k', linestyle='dashdot', lw=1)
            ax.axvline(x=x[idx][endTmp[i]], color='orange', linestyle='dashed', lw=1)
            ax.axvline(x=x[idx][end[i]], color=[70 / 255, 130 / 255, 180 / 255], linestyle='solid', lw=1)
            if i:
                ax.plot(np.linspace(x[idx][end[i-1]], x[idx][endTmp[i]], 100), [ITUTmp[i]] * 100, color='orange',
                        linestyle='dashed', lw=1)
                ax.fill_between(np.linspace(x[idx][end[i-1]], x[idx][start[i]], 100), -1000, [500] * 100,
                                facecolor=[0, 1, 1],
                                alpha=0.5)
    elif idx == 2:
        # ax.set_yticks(np.array(range(-2000, 2500, 1000)))
        for i in range(len(start)):
            ax.axvspan(x[idx][start[i]], x[idx][startTmp[i]], facecolor=[84 / 255, 1, 159 / 255], alpha=0.5)
            ax.axvline(x=x[idx][startTmp[i]], color='k', linestyle='dashdot', lw=1)
    else:
        # ax.set_yticks(np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]))
        ax.axvspan(0, t_backNoise, facecolor='orange', alpha=0.2)
        for i in range(len(start)):
            ax.axvspan(x[idx][endTmp[i]], x[idx][end[i]], facecolor=[84 / 255, 1, 159 / 255], alpha=0.5)
            ax.axvline(x=x[idx][endTmp[i]], color='orange', linestyle='dashed', lw=1)
            ax.axvline(x=x[idx][end[i]], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
            ax.plot(np.linspace(x[idx][endTmp[i]], x[idx][end[i]], 100), [IZCRTTmp[i]] * 100,
                    color=[70 / 255, 130 / 255, 180 / 255],
                    linestyle='dashed', lw=1)
            if i:
                ax.axvspan(x[idx][end[i-1]], x[idx][start[i]], facecolor=[0, 1, 1], alpha=0.5)

    ax.grid(linewidth=0.3)
    plot_norm(ax, 'Time / μs' if idx == 3 else '', ylabel[idx], legend=False, labelWeight='normal')

plt.subplots_adjust(wspace=0, hspace=0)

# ----------------------------------------------------- Alpha 1.5 -----------------------------------------------------
staLen, overlap, staWin, IZCRT, ITU, alpha, t_backNoise = 2, 1, 'hamming', 0.3, 650, 1.5, 50
t_str, t_end = 50, 1e7
t = t_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1] - t_initial[int(t_str // t_initial[1])]
sig = sig_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1]

width = int(20 * staLen)
stride = int(width) - overlap
t_stE, stE, zcR = shortTermEny_zerosCrossingRate(sig, width, stride, 20, staWin)
stE_dev = cal_deriv(t_stE, stE)
# start, end = find_wave(stE, stE_dev, zcR, t_stE, IZCRT=IZCRT, ITU=ITU, alpha=alpha, t_backNoise=t_backNoise)
start, end, startTmp, endTmp, ITUTmp, IZCRTTmp, ITLTmp = find_wave_multiOutput(stE, stE_dev, zcR, t_stE, IZCRT=IZCRT,
                                                                               ITU=ITU, alpha=alpha,
                                                                               t_backNoise=t_backNoise)

x = [t, t_stE, t_stE, t_stE]
y = [sig, stE, stE_dev, zcR]
color = ['black', 'green', 'gray', 'purple']
ylabel = ['Amplitude / μV', r'STEnergy / μV$^2 \cdot$μs', r'S$\dot{T}$E / μV$^2$', r'ST$\widehat{Z}$CR / %']
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(5, 8))
for idx, ax in enumerate(axes):
    ax.plot(x[idx], y[idx], lw=1, color=color[idx])
    if idx == 0:
        for s, e in tqdm(zip(start, end)):
            ax.plot(t[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2],
                    sig[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2], lw=1, color='red')
            print(t[int(t_stE[s] // t[1]) + 1], t[int(t_stE[e] // t[1]) + 1],
                  t[int(t_stE[e] // t[1]) + 1] - t[int(t_stE[s] // t[1]) + 1])
            ax.axvspan(t[int(t_stE[s] // t[1]) + 1], t[int(t_stE[e] // t[1]) + 1], facecolor=[84 / 255, 1, 159 / 255],
                       alpha=0.5)
            ax.axvline(x=t[int(t_stE[s] // t[1]) + 1], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
            ax.axvline(x=t[int(t_stE[e] // t[1]) + 1], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
        # ax.set_xticks(np.array(range(0, 251, 25)))
    elif idx == 1:
        # ax.set_yticks(np.array(range(-1000, 10000, 2500)))
        ax.fill_between(np.linspace(0, t_backNoise, 10), -1000, [500] * 10, facecolor='orange', alpha=0.2)
        ax.fill_between(x[idx], -1000, y[idx], facecolor=[95 / 255, 158 / 255, 160 / 255], alpha=0.2)
        ax.plot(np.linspace(t_backNoise, x[idx][endTmp[0]], 100), [ITUTmp[0]] * 100, color='orange', linestyle='dashed', lw=1)
        for i in range(len(start)):
            ax.axvspan(x[idx][startTmp[i]], x[idx][endTmp[i]], facecolor=[95 / 255, 158 / 255, 160 / 255], alpha=0.2)
            ax.fill_between(x[idx][startTmp[i]:endTmp[i] + 1], -1000, y[idx][startTmp[i]:endTmp[i] + 1],
                            facecolor=[28 / 255, 28 / 255, 28 / 255], alpha=0.5)
            ax.axvline(x=x[idx][startTmp[i]], color='k', linestyle='dashdot', lw=1)
            ax.axvline(x=x[idx][endTmp[i]], color='orange', linestyle='dashed', lw=1)
            ax.axvline(x=x[idx][end[i]], color=[70 / 255, 130 / 255, 180 / 255], linestyle='solid', lw=1)
            if i:
                ax.plot(np.linspace(x[idx][end[i-1]], x[idx][endTmp[i]], 100), [ITUTmp[i]] * 100, color='orange',
                        linestyle='dashed', lw=1)
                ax.fill_between(np.linspace(x[idx][end[i-1]], x[idx][start[i]], 100), -1000, [500] * 100,
                                facecolor=[0, 1, 1],
                                alpha=0.5)
    elif idx == 2:
        # ax.set_yticks(np.array(range(-2000, 2500, 1000)))
        for i in range(len(start)):
            ax.axvspan(x[idx][start[i]], x[idx][startTmp[i]], facecolor=[84 / 255, 1, 159 / 255], alpha=0.5)
            ax.axvline(x=x[idx][startTmp[i]], color='k', linestyle='dashdot', lw=1)
    else:
        # ax.set_yticks(np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]))
        ax.axvspan(0, t_backNoise, facecolor='orange', alpha=0.2)
        for i in range(len(start)):
            ax.axvspan(x[idx][endTmp[i]], x[idx][end[i]], facecolor=[84 / 255, 1, 159 / 255], alpha=0.5)
            ax.axvline(x=x[idx][endTmp[i]], color='orange', linestyle='dashed', lw=1)
            ax.axvline(x=x[idx][end[i]], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
            ax.plot(np.linspace(x[idx][endTmp[i]], x[idx][end[i]], 100), [IZCRTTmp[i]] * 100,
                    color=[70 / 255, 130 / 255, 180 / 255],
                    linestyle='dashed', lw=1)
            if i:
                ax.axvspan(x[idx][end[i-1]], x[idx][start[i]], facecolor=[0, 1, 1], alpha=0.5)

    ax.grid(linewidth=0.3)
    plot_norm(ax, 'Time / μs' if idx == 3 else '', ylabel[idx], legend=False, labelWeight='normal')

plt.subplots_adjust(wspace=0, hspace=0)


# ==================================================== Same Alpha ======================================================
t_initial, sig_initial = waveform.cal_wave(data_tra[185944 - 1], False)

# ------------------------------------------------------ ITU 650 -------------------------------------------------------
staLen, overlap, staWin, IZCRT, ITU, alpha, t_backNoise = 2, 10, 'hamming', 0.2, 650, 1.3, 0
t_str, t_end = 0, 1e7
t = t_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1] - t_initial[int(t_str // t_initial[1])]
sig = sig_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1]

width = int(20 * staLen)
stride = int(width) - overlap
t_stE, stE, zcR = shortTermEny_zerosCrossingRate(sig, width, stride, 20, staWin)
stE *= 1.2
stE_dev = cal_deriv(t_stE, stE)
# start, end = find_wave(stE, stE_dev, zcR, t_stE, IZCRT=IZCRT, ITU=ITU, alpha=alpha, t_backNoise=t_backNoise)
start, end, startTmp, endTmp, ITUTmp, IZCRTTmp, ITLTmp = find_wave_multiOutput(stE, stE_dev, zcR, t_stE, IZCRT=IZCRT,
                                                                               ITU=ITU, alpha=alpha,
                                                                               t_backNoise=t_backNoise)

x = [t, t_stE, t_stE, t_stE]
y = [sig, stE, stE_dev, zcR]
color = ['black', 'green', 'gray', 'purple']
ylabel = ['Amplitude / μV', r'STEnergy / μV$^2 \cdot$μs', r'S$\dot{T}$E / μV$^2$', r'ST$\widehat{Z}$CR / %']
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(5, 8))
for idx, ax in enumerate(axes):
    ax.plot(x[idx], y[idx], lw=1, color=color[idx])
    if idx == 0:
        for s, e in tqdm(zip(start, end)):
            ax.plot(t[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2],
                    sig[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2], lw=1, color='red')
            print(t[int(t_stE[s] // t[1]) + 1], t[int(t_stE[e] // t[1]) + 1],
                  t[int(t_stE[e] // t[1]) + 1] - t[int(t_stE[s] // t[1]) + 1])
            ax.axvspan(t[int(t_stE[s] // t[1]) + 1], t[int(t_stE[e] // t[1]) + 1], facecolor=[84 / 255, 1, 159 / 255],
                       alpha=0.5)
            ax.axvline(x=t[int(t_stE[s] // t[1]) + 1], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
            ax.axvline(x=t[int(t_stE[e] // t[1]) + 1], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
        # ax.set_xticks(np.array(range(0, 251, 25)))
    elif idx == 1:
        # ax.set_yticks(np.array(range(-1000, 10000, 2500)))
        ax.fill_between(np.linspace(0, t_backNoise, 10), -1000, [500] * 10, facecolor='orange', alpha=0.2)
        ax.fill_between(x[idx], -1000, y[idx], facecolor=[95 / 255, 158 / 255, 160 / 255], alpha=0.2)
        ax.plot(np.linspace(t_backNoise, x[idx][endTmp[0]], 100), [ITUTmp[0]] * 100, color='orange', linestyle='dashed', lw=1)
        for i in range(len(start)):
            ax.axvspan(x[idx][startTmp[i]], x[idx][endTmp[i]], facecolor=[95 / 255, 158 / 255, 160 / 255], alpha=0.2)
            ax.fill_between(x[idx][startTmp[i]:endTmp[i] + 1], -1000, y[idx][startTmp[i]:endTmp[i] + 1],
                            facecolor=[28 / 255, 28 / 255, 28 / 255], alpha=0.5)
            ax.axvline(x=x[idx][startTmp[i]], color='k', linestyle='dashdot', lw=1)
            ax.axvline(x=x[idx][endTmp[i]], color='orange', linestyle='dashed', lw=1)
            ax.axvline(x=x[idx][end[i]], color=[70 / 255, 130 / 255, 180 / 255], linestyle='solid', lw=1)
            if i:
                ax.plot(np.linspace(x[idx][end[i-1]], x[idx][endTmp[i]], 100), [ITUTmp[i]] * 100, color='orange',
                        linestyle='dashed', lw=1)
                ax.fill_between(np.linspace(x[idx][end[i-1]], x[idx][start[i]], 100), -1000, [500] * 100,
                                facecolor=[0, 1, 1],
                                alpha=0.5)
    elif idx == 2:
        # ax.set_yticks(np.array(range(-2000, 2500, 1000)))
        for i in range(len(start)):
            ax.axvspan(x[idx][start[i]], x[idx][startTmp[i]], facecolor=[84 / 255, 1, 159 / 255], alpha=0.5)
            ax.axvline(x=x[idx][startTmp[i]], color='k', linestyle='dashdot', lw=1)
    else:
        # ax.set_yticks(np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]))
        ax.axvspan(0, t_backNoise, facecolor='orange', alpha=0.2)
        for i in range(len(start)):
            ax.axvspan(x[idx][endTmp[i]], x[idx][end[i]], facecolor=[84 / 255, 1, 159 / 255], alpha=0.5)
            ax.axvline(x=x[idx][endTmp[i]], color='orange', linestyle='dashed', lw=1)
            ax.axvline(x=x[idx][end[i]], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
            ax.plot(np.linspace(x[idx][endTmp[i]], x[idx][end[i]], 100), [IZCRTTmp[i]] * 100,
                    color=[70 / 255, 130 / 255, 180 / 255],
                    linestyle='dashed', lw=1)
            if i:
                ax.axvspan(x[idx][end[i-1]], x[idx][start[i]], facecolor=[0, 1, 1], alpha=0.5)

    ax.grid(linewidth=0.3)
    plot_norm(ax, 'Time / μs' if idx == 3 else '', ylabel[idx], legend=False, labelWeight='normal')

plt.subplots_adjust(wspace=0, hspace=0)

# ------------------------------------------------------ ITU 750 -------------------------------------------------------
staLen, overlap, staWin, IZCRT, ITU, alpha, t_backNoise = 2, 10, 'hamming', 0.2, 750, 1.3, 0
t_str, t_end = 0, 1e7
t = t_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1] - t_initial[int(t_str // t_initial[1])]
sig = sig_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1]

width = int(20 * staLen)
stride = int(width) - overlap
t_stE, stE, zcR = shortTermEny_zerosCrossingRate(sig, width, stride, 20, staWin)
stE *= 1.2
stE_dev = cal_deriv(t_stE, stE)
# start, end = find_wave(stE, stE_dev, zcR, t_stE, IZCRT=IZCRT, ITU=ITU, alpha=alpha, t_backNoise=t_backNoise)
start, end, startTmp, endTmp, ITUTmp, IZCRTTmp, ITLTmp = find_wave_multiOutput(stE, stE_dev, zcR, t_stE, IZCRT=IZCRT,
                                                                               ITU=ITU, alpha=alpha,
                                                                               t_backNoise=t_backNoise)

x = [t, t_stE, t_stE, t_stE]
y = [sig, stE, stE_dev, zcR]
color = ['black', 'green', 'gray', 'purple']
ylabel = ['Amplitude / μV', r'STEnergy / μV$^2 \cdot$μs', r'S$\dot{T}$E / μV$^2$', r'ST$\widehat{Z}$CR / %']
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(5, 8))
for idx, ax in enumerate(axes):
    ax.plot(x[idx], y[idx], lw=1, color=color[idx])
    if idx == 0:
        for s, e in tqdm(zip(start, end)):
            ax.plot(t[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2],
                    sig[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2], lw=1, color='red')
            print(t[int(t_stE[s] // t[1]) + 1], t[int(t_stE[e] // t[1]) + 1],
                  t[int(t_stE[e] // t[1]) + 1] - t[int(t_stE[s] // t[1]) + 1])
            ax.axvspan(t[int(t_stE[s] // t[1]) + 1], t[int(t_stE[e] // t[1]) + 1], facecolor=[84 / 255, 1, 159 / 255],
                       alpha=0.5)
            ax.axvline(x=t[int(t_stE[s] // t[1]) + 1], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
            ax.axvline(x=t[int(t_stE[e] // t[1]) + 1], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
        # ax.set_xticks(np.array(range(0, 251, 25)))
    elif idx == 1:
        # ax.set_yticks(np.array(range(-1000, 10000, 2500)))
        ax.fill_between(np.linspace(0, t_backNoise, 10), -1000, [500] * 10, facecolor='orange', alpha=0.2)
        ax.fill_between(x[idx], -1000, y[idx], facecolor=[95 / 255, 158 / 255, 160 / 255], alpha=0.2)
        ax.plot(np.linspace(t_backNoise, x[idx][endTmp[0]], 100), [ITUTmp[0]] * 100, color='orange', linestyle='dashed', lw=1)
        for i in range(len(start)):
            ax.axvspan(x[idx][startTmp[i]], x[idx][endTmp[i]], facecolor=[95 / 255, 158 / 255, 160 / 255], alpha=0.2)
            ax.fill_between(x[idx][startTmp[i]:endTmp[i] + 1], -1000, y[idx][startTmp[i]:endTmp[i] + 1],
                            facecolor=[28 / 255, 28 / 255, 28 / 255], alpha=0.5)
            ax.axvline(x=x[idx][startTmp[i]], color='k', linestyle='dashdot', lw=1)
            ax.axvline(x=x[idx][endTmp[i]], color='orange', linestyle='dashed', lw=1)
            ax.axvline(x=x[idx][end[i]], color=[70 / 255, 130 / 255, 180 / 255], linestyle='solid', lw=1)
            if i:
                ax.plot(np.linspace(x[idx][end[i-1]], x[idx][endTmp[i]], 100), [ITUTmp[i]] * 100, color='orange',
                        linestyle='dashed', lw=1)
                ax.fill_between(np.linspace(x[idx][end[i-1]], x[idx][start[i]], 100), -1000, [500] * 100,
                                facecolor=[0, 1, 1],
                                alpha=0.5)
    elif idx == 2:
        # ax.set_yticks(np.array(range(-2000, 2500, 1000)))
        for i in range(len(start)):
            ax.axvspan(x[idx][start[i]], x[idx][startTmp[i]], facecolor=[84 / 255, 1, 159 / 255], alpha=0.5)
            ax.axvline(x=x[idx][startTmp[i]], color='k', linestyle='dashdot', lw=1)
    else:
        # ax.set_yticks(np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]))
        ax.axvspan(0, t_backNoise, facecolor='orange', alpha=0.2)
        for i in range(len(start)):
            ax.axvspan(x[idx][endTmp[i]], x[idx][end[i]], facecolor=[84 / 255, 1, 159 / 255], alpha=0.5)
            ax.axvline(x=x[idx][endTmp[i]], color='orange', linestyle='dashed', lw=1)
            ax.axvline(x=x[idx][end[i]], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
            ax.plot(np.linspace(x[idx][endTmp[i]], x[idx][end[i]], 100), [IZCRTTmp[i]] * 100,
                    color=[70 / 255, 130 / 255, 180 / 255],
                    linestyle='dashed', lw=1)
            if i:
                ax.axvspan(x[idx][end[i-1]], x[idx][start[i]], facecolor=[0, 1, 1], alpha=0.5)

    ax.grid(linewidth=0.3)
    plot_norm(ax, 'Time / μs' if idx == 3 else '', ylabel[idx], legend=False, labelWeight='normal')

plt.subplots_adjust(wspace=0, hspace=0)

# ------------------------------------------------------ ITU 1250 -------------------------------------------------------
staLen, overlap, staWin, IZCRT, ITU, alpha, t_backNoise = 2, 10, 'hamming', 0.2, 1250, 1.3, 0
t_str, t_end = 0, 1e7
t = t_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1] - t_initial[int(t_str // t_initial[1])]
sig = sig_initial[int(t_str // t_initial[1]):int(t_end // t_initial[1]) + 1]

width = int(20 * staLen)
stride = int(width) - overlap
t_stE, stE, zcR = shortTermEny_zerosCrossingRate(sig, width, stride, 20, staWin)
stE *= 1.2
stE_dev = cal_deriv(t_stE, stE)
# start, end = find_wave(stE, stE_dev, zcR, t_stE, IZCRT=IZCRT, ITU=ITU, alpha=alpha, t_backNoise=t_backNoise)
start, end, startTmp, endTmp, ITUTmp, IZCRTTmp, ITLTmp = find_wave_multiOutput(stE, stE_dev, zcR, t_stE, IZCRT=IZCRT,
                                                                               ITU=ITU, alpha=alpha,
                                                                               t_backNoise=t_backNoise)

x = [t, t_stE, t_stE, t_stE]
y = [sig, stE, stE_dev, zcR]
color = ['black', 'green', 'gray', 'purple']
ylabel = ['Amplitude / μV', r'STEnergy / μV$^2 \cdot$μs', r'S$\dot{T}$E / μV$^2$', r'ST$\widehat{Z}$CR / %']
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(5, 8))
for idx, ax in enumerate(axes):
    ax.plot(x[idx], y[idx], lw=1, color=color[idx])
    if idx == 0:
        for s, e in tqdm(zip(start, end)):
            ax.plot(t[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2],
                    sig[int(t_stE[s] // t[1]) + 1:int(t_stE[e] // t[1]) + 2], lw=1, color='red')
            print(t[int(t_stE[s] // t[1]) + 1], t[int(t_stE[e] // t[1]) + 1],
                  t[int(t_stE[e] // t[1]) + 1] - t[int(t_stE[s] // t[1]) + 1])
            ax.axvspan(t[int(t_stE[s] // t[1]) + 1], t[int(t_stE[e] // t[1]) + 1], facecolor=[84 / 255, 1, 159 / 255],
                       alpha=0.5)
            ax.axvline(x=t[int(t_stE[s] // t[1]) + 1], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
            ax.axvline(x=t[int(t_stE[e] // t[1]) + 1], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
        # ax.set_xticks(np.array(range(0, 251, 25)))
    elif idx == 1:
        # ax.set_yticks(np.array(range(-1000, 10000, 2500)))
        ax.fill_between(np.linspace(0, t_backNoise, 10), -1000, [500] * 10, facecolor='orange', alpha=0.2)
        ax.fill_between(x[idx], -1000, y[idx], facecolor=[95 / 255, 158 / 255, 160 / 255], alpha=0.2)
        ax.plot(np.linspace(t_backNoise, x[idx][endTmp[0]], 100), [ITUTmp[0]] * 100, color='orange', linestyle='dashed', lw=1)
        for i in range(len(start)):
            ax.axvspan(x[idx][startTmp[i]], x[idx][endTmp[i]], facecolor=[95 / 255, 158 / 255, 160 / 255], alpha=0.2)
            ax.fill_between(x[idx][startTmp[i]:endTmp[i] + 1], -1000, y[idx][startTmp[i]:endTmp[i] + 1],
                            facecolor=[28 / 255, 28 / 255, 28 / 255], alpha=0.5)
            ax.axvline(x=x[idx][startTmp[i]], color='k', linestyle='dashdot', lw=1)
            ax.axvline(x=x[idx][endTmp[i]], color='orange', linestyle='dashed', lw=1)
            ax.axvline(x=x[idx][end[i]], color=[70 / 255, 130 / 255, 180 / 255], linestyle='solid', lw=1)
            if i:
                ax.plot(np.linspace(x[idx][end[i-1]], x[idx][endTmp[i]], 100), [ITUTmp[i]] * 100, color='orange',
                        linestyle='dashed', lw=1)
                ax.fill_between(np.linspace(x[idx][end[i-1]], x[idx][start[i]], 100), -1000, [500] * 100,
                                facecolor=[0, 1, 1],
                                alpha=0.5)
    elif idx == 2:
        # ax.set_yticks(np.array(range(-2000, 2500, 1000)))
        for i in range(len(start)):
            ax.axvspan(x[idx][start[i]], x[idx][startTmp[i]], facecolor=[84 / 255, 1, 159 / 255], alpha=0.5)
            ax.axvline(x=x[idx][startTmp[i]], color='k', linestyle='dashdot', lw=1)
    else:
        # ax.set_yticks(np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]))
        ax.axvspan(0, t_backNoise, facecolor='orange', alpha=0.2)
        for i in range(len(start)):
            ax.axvspan(x[idx][endTmp[i]], x[idx][end[i]], facecolor=[84 / 255, 1, 159 / 255], alpha=0.5)
            ax.axvline(x=x[idx][endTmp[i]], color='orange', linestyle='dashed', lw=1)
            ax.axvline(x=x[idx][end[i]], color=[78 / 255, 238 / 255, 148 / 255], linestyle='solid', lw=1)
            ax.plot(np.linspace(x[idx][endTmp[i]], x[idx][end[i]], 100), [IZCRTTmp[i]] * 100,
                    color=[70 / 255, 130 / 255, 180 / 255],
                    linestyle='dashed', lw=1)
            if i:
                ax.axvspan(x[idx][end[i-1]], x[idx][start[i]], facecolor=[0, 1, 1], alpha=0.5)

    ax.grid(linewidth=0.3)
    plot_norm(ax, 'Time / μs' if idx == 3 else '', ylabel[idx], legend=False, labelWeight='normal')

plt.subplots_adjust(wspace=0, hspace=0)
