import sys
import numpy as np
import scipy.io.wavfile
import scipy.special
import matplotlib.pyplot as plt

def chi2_dim1_cdf(x):
    ''' Cumulative probability distribution function of chi-squared distribution with 1 dimension,
        i.e. distribution of power if the signal is unity normal distributed.
    '''
    return scipy.special.erf(np.sqrt(x / 2 ))

def goertzel_power(frame, k):
    π = np.pi
    ω = 2 * π * k / len(frame)
    coeff = 2 * np.cos(ω)

    sprev = 0
    sprev2 = 0
    for x_n in frame:
        s = x_n + (coeff * sprev) - sprev2
        sprev2 = sprev
        sprev = s

    power = (sprev2 * sprev2) + (sprev * sprev) - (coeff * sprev * sprev2)
    return power

def main():
    fs, sig = scipy.io.wavfile.read(sys.argv[1])
    sig = sig * (1/32768)

    fdet = int(sys.argv[2])

    N = 120
    S = N//5
    K = int(0.5 + (N * fdet / fs))
    print(f'Sampling frequency: {fs}Hz')
    print(f'Detection frequency: {K*fs/N}Hz')
    win = np.hamming(N)

    y = list()
    for idx in range(0, len(sig) - N, S):
        frame = win * sig[idx:idx+N]
        power1 = goertzel_power(frame, K)
        power2 = goertzel_power(frame, K + 2)
        power3 = goertzel_power(frame, K - 2)
        y.append( (power1, power2, power3) )

    y = np.array(y)

    #sigma = (np.std(y[:,1]) + np.std(y[:,2])) / 2
    sigma = min(np.mean(y, axis=0))
    p = chi2_dim1_cdf(y[:,0] / (3*sigma))

    print(np.mean(y, axis=0), np.std(y, axis=0), sigma)

    #plt.plot(y/sigma / 6)
    # p3 = p[:-1] * p[1]
    # p3 = p[:-2] * p[1:-1] * p[2:]
    p3 = p[:-3] * p[1:-2] * p[2:-1] * p[3:]

    key = False
    all_stats = list()
    key_count = 0
    p3_onoff = list()
    for p_n in p3:
        if not key and p_n > 0.7:
            key = True
            all_stats.append(key_count)
            key_count = 0
        elif key and p_n < 0.3:
            key = False
            all_stats.append(key_count)
            key_count = 0

        if key:
            key_count += 1
        else:
            key_count -= 1
        
        p3_onoff.append(key)

    print(all_stats)
    key_stats = [x_n for x_n in all_stats if x_n > 0]
    dah_threshold = np.mean(key_stats)
    is_dah = np.array(key_stats) > dah_threshold
    k_time = (is_dah / 3) + (1 - is_dah)
    key_stats = k_time * key_stats
    len_unit = np.mean(k_time * key_stats)

    t_unit = len_unit / (fs/S)
    wpm = 1.2 / t_unit
    print(f'len_unit = {len_unit:.1f}')
    print(f't_unit = {t_unit:.3f}s')
    print(f'WPM = {wpm:.1f}')

    decode = ''
    for k in all_stats:
        if k > 0.5 * len_unit and k < 2.2 * len_unit:
        # if k > 0 and k < 2.2 * len_unit:
            decode += '.'
        # if k > 2 * len_unit and k < 4 * len_unit:
        if k > 2.2 * len_unit:
            decode += '-'
        if k < -2.5 * len_unit and k > -5 * len_unit:
            decode += ' '
        if k < -5 * len_unit and k > -8 * len_unit:
            decode += ' / '
        if k < -8 * len_unit:
            decode += '\n'

    print(decode)

    #plt.plot(p3)
    p3_disp = np.concatenate( [np.tile(p3, (200, 1)), np.tile(p3_onoff, (200, 1))] )
    plt.imshow(p3_disp, vmin=0, vmax=1, cmap=plt.get_cmap('viridis'))
    plt.show()

if __name__ == '__main__':
    main()
    sys.exit(0)
