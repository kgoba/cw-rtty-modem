import matplotlib.pyplot as plt
import scipy.io.wavfile
import scipy.signal
import scipy.special
import numpy as np

MIN_SYMBOL_LENGTH = 0.028125 # 30 ms is for 40 WPM
SYMBOL_OSR = 3 # try to get at least 3 samples per symbol (necessary for stable FST operation)

def zeta(theta):
    t2 = theta**2
    return 2 + t2 - np.pi/8 * np.exp(-t2 / 2) * ((2 + t2) * scipy.special.iv(0, t2/4) + t2 * scipy.special.iv(1, t2/4))**2

def analyse1(fgrid, tgrid, A):

    k1 = np.sqrt(np.var(A, axis=1)/0.429) # Var(Rayleigh) = sigma^2 * (4-pi)/2
    k2 = np.mean(A, axis=1)/1.253         # Mean(Rayleigh) = sigma * sqrt(pi/2)
    k3 = np.sqrt(np.mean(A**2, axis=1)/2) # sigma_mle(Rayleigh)
    r = np.log(k1 / (1e-4 + k2))
    peak_idx = np.argmax(k1[2:-2]) + 2

    print(f'Found peak at {fgrid[peak_idx]:.1f} Hz')

    #sigma_nse = np.sqrt(0.9*np.mean(np.mean(A[peak_idx + 10:peak_idx + 30, :], axis=0)**2)/2)
    sigma_nse = np.median(k3[peak_idx - 20: peak_idx + 21])
    print(f'Noise sigma estimate = {sigma_nse:.5f}')
    # plt.plot(fgrid, r)
    # plt.plot( (fgrid[peak_idx], fgrid[peak_idx]), (-1, 1))

    Ap = A[peak_idx, :]
    Ap = np.convolve(Ap, [1.0/3] * 3, 'same')

    Apenv = []
    for idx in range(-5, 30+1):
        # Apenv.append(np.roll(Ap, idx) * 1 / (1 + 0.0002 * idx*idx))
        Apenv.append(np.roll(Ap, idx) * np.exp(-0.0002 * idx*idx))
    Apenv = np.max(Apenv, axis=0)

    nse_floor = 2 * sigma_nse
    det = np.where(Ap > nse_floor, Ap - nse_floor, 0)
    det = det / (Apenv - nse_floor)

    # plt.plot(tgrid, Ap, marker='.')
    # plt.plot(tgrid, Apenv, marker='.')
    plt.plot(tgrid, det, marker='.')
    # plt.plot(tgrid, np.convolve(Ap, [1.0/3] * 3, 'same'), marker='.')
    # plt.plot(tgrid, A[peak_idx - 5, :], marker='.')
    # plt.plot((tgrid[0], tgrid[-1]), (sigma_nse, sigma_nse))
    # plt.plot((tgrid[0], tgrid[-1]), (2*sigma_nse, 2*sigma_nse))
    # plt.plot(tgrid, np.where(Ap > 2*sigma_nse, 20*np.log10(Ap / (2*sigma_nse)), 0), marker='.')
    plt.grid()
    plt.figure()
    # plt.hist(Ap, bins=50)
    # plt.hist(np.sqrt(np.mean(A**2, axis=1)/2), bins=50)
    plt.plot(fgrid, np.sqrt(np.mean(A**2, axis=1)/2), marker='.')
    plt.plot(fgrid, sorted(np.sqrt(np.mean(A**2, axis=1)/2)))
    # prev_peak = None
    # selected_peak = None
    # peak_timer = 0

    # step = A.shape[1]//10
    # for idx in range(0, A.shape[1], step):
    #     Awin = A[:, idx:idx+step]
    #     k1 = np.sqrt(np.var(Awin, axis=1)/0.429) # Var(Rayleigh) = sigma^2 * (4-pi)/2
    #     k2 = np.mean(Awin, axis=1)/1.253         # Mean(Rayleigh) = sigma * sqrt(pi/2)
    #     k3 = np.sqrt(np.mean(Awin**2, axis=1)/2) # MLE(sigma^2) = 1/2 * sum(x_i^2) / N

    #     r = np.log(k1 / (1e-4 + k2))
    #     curr_peak = np.argmax(r)
    #     if r[curr_peak] > 0.5:
    #         print(f'curr_peak = {curr_peak} ({fgrid[curr_peak]:.1f} Hz)')
    #         if curr_peak != prev_peak:
    #             prev_peak = curr_peak
    #             peak_timer = 0
    #         else:
    #             if peak_timer < 3:
    #                 peak_timer += 1
    #             else:
    #                 print(f'{tgrid[idx]:.3f}: Switching peak: {selected_peak} -> {curr_peak} ({fgrid[curr_peak]:.1f} Hz)')
    #                 selected_peak = curr_peak

    #     # m2 = np.mean(Awin**2, axis=1)
    #     # m4 = np.mean(Awin**4, axis=1)
    #     # tmp = (2 * m2*m2 - m4)
    #     # tmp = np.where(tmp > 0, tmp, 0)
    #     # mu_c = np.sqrt(np.sqrt(tmp))
    #     # mu = np.sqrt(m2 - np.sqrt(tmp))

    #     # r = np.mean(Awin, axis=1) / np.std(Awin, axis=1, ddof=1)
    #     # theta = (10.0)* np.ones(Awin.shape[0])
    #     # for idx in range(10):
    #     #     theta_new = np.sqrt(zeta(theta) * (1 + r**2) - 2)
    #     #     print(theta_new - theta)
    #     #     theta = theta_new

    #     # plt.plot(fgrid, np.sqrt(np.mean(A**2, axis=1)/2)*1.2, marker='.')
    #     # plt.plot(fgrid, k1, marker='.')
    #     # plt.plot(fgrid, k2, marker='.')
    #     # plt.plot(fgrid, k3, marker='.')
    #     plt.plot(fgrid, r, marker='.')
    #     # plt.plot(fgrid, mu_c)
    #     # plt.plot(fgrid, mu)
    #     # plt.plot(fgrid, np.log(k1 / (1e-4 + k2)), marker='.')
    #     # plt.plot(fgrid, np.std(A**2, axis=1), marker='.')
    #     # plt.plot(fgrid, np.mean(A**2, axis=1), marker='.')

def analyse2(fgrid, tgrid, H):
    win_size = 256
    win_shift = 32
    fs = 1 / (MIN_SYMBOL_LENGTH / SYMBOL_OSR)
    A = np.abs(H)
    A2 = list()
    for bin in range(A.shape[0]):
        sig2 = scipy.signal.lfilter([1, -1], 1, A[bin, :])
        _, _, H2 = scipy.signal.stft(sig2 - np.mean(sig2), window='boxcar', nperseg=win_size, noverlap=win_size - win_shift, nfft=win_size, fs=fs)
        # A2.append(np.mean(np.abs(H2[15:90, :]), axis=0))
        A2.append(np.abs(H2))
    plt.plot(np.mean(A2, axis=(0, 2)))
    # plt.plot(np.mean(A2, axis=1))
    # plt.plot(A2)
    # A2_env = scipy.signal.convolve2d(A2, [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]], mode='same', boundary='wrap')
    # A2_env = scipy.signal.filtfilt([1, 1, 1], 1, np.mean(A2, axis=0)) / 9
    # A2_env = np.max(A2, axis=0)
    A2_env = 1
    # plt.imshow(np.array(A2) / A2_env)

def main():
    fs, sig = scipy.io.wavfile.read(sys.argv[1])
    sig = sig * (1.0/32768)
    print(f'Fs={fs} Hz, file length {len(sig)/fs:.3f} s')
    sym_size = int(0.5 + MIN_SYMBOL_LENGTH * fs)
    win_size = sym_size
    win_shift = sym_size // SYMBOL_OSR
    nfft = win_size
    fgrid, tgrid, H = scipy.signal.stft(sig, window='hamming', nperseg=win_size, noverlap=win_size - win_shift, nfft=nfft, fs=fs)

    print(f'Analysis window size {win_size} samples ({fs/nfft:.1f} Hz per bin)')

    analyse2(fgrid, tgrid, H)

    # plt.grid()
    plt.show()

if __name__ == '__main__':
    import sys
    main()
    sys.exit(0)