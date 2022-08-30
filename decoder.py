import sys
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

from morse import MORSE_TABLE
from stats import *
from dsp import *
from fst import *

class CWMonitor:
    def __init__(self, fs, fmin=300, fmax=3000):
        self.fs = fs
        self.N = int(fs * 0.060)
        self.L = self.N // 2
        self.win = np.blackman(self.N)
        self.buf = []
        self.idx_lo = int(self.N * fmin / fs)
        self.idx_hi = int(self.N * fmax / fs)
        self.fmin = fmin
        self.hist = []

    def process(self, samples):
        self.buf.extend(samples)
        pos = 0
        while pos + self.N < len(self.buf):
            self.process_block(self.buf[pos:pos+self.N])
            pos += self.L
        self.buf = self.buf[pos:]

    def process_block(self, block):
        H = np.fft.rfft(self.win * block)[self.idx_lo:self.idx_hi]
        A = np.abs(H)
        Adb = 20*np.log10(A + 1e-6)
        if len(self.hist) > 30:
            self.hist = self.hist[-30:]
        self.hist.append(Adb)

        # V = np.std(self.hist, axis=0)
        # idx = np.argmax(V)
        # freq = (self.idx_lo + idx) * self.fs / self.N
        # print(idx, freq, V[idx])

class CWFrontend:
    def __init__(self, N, K, D):
        self.N = N
        self.K = K
        self.D = D
        # self.win = np.hamming(N)
        self.win = np.blackman(N)
        self.sigma_nse = 0.05
        self.sigma_sig = 0.2
        self.p_stat = []
        self.sig_stat = []
        self.sig_stat_norm = []
        self.noise_stat = []
        self.p_hist = [0.1, 0.1, 0.1, 0.1]
        self.last_power = 0

    def process(self, frame):
        scale = 1.0 / self.N
        frame_win = self.win * (frame) # + 0.2 * np.random.normal(size=self.N))
        power1 = scale * goertzel_power(frame_win, self.K)
        power2 = scale * goertzel_power(frame_win, self.K + 3)
        power3 = scale * goertzel_power(frame_win, self.K - 3)
        power4 = scale * goertzel_power(frame_win, self.K + 2)
        power5 = scale * goertzel_power(frame_win, self.K - 2)

        power1 += 0.07 * (power1 - self.last_power)
        if power1 < 1e-9:
            power1 = 1e-9
        self.last_power = power1

        # noise_pwr = 3.4 * min( (power1, power2, power3, power4, power5) )
        # noise_pwr = 2.7 * min( (power1, power2, power3) )
        # noise_pwr = np.mean( (power2, power3) )
        noise_pwr = np.mean( (power2, power3, power4, power5) )
        self.sigma_nse += 0.1 * (noise_pwr - self.sigma_nse)

        # Calculate posterior probability from two component mixture pdf
        pd_noise = pdf_half_normal(np.sqrt(power1), np.sqrt(self.sigma_nse))
        # pd_noise = pdf_chi2_1dim(power1 / self.sigma_nse, 1)
        pd_signal = pdf_rayleigh(np.sqrt(power1), np.sqrt(self.sigma_sig/2))
        # pd_signal = pdf_rayleigh(np.sqrt(power1) / self.sigma_sig / 1.0, 1.0)
        p = pd_signal / (pd_noise + pd_signal) # mixture weights are assumed 0.5/0.5

        # if power1 > 9 * self.sigma_nse:
        if p > 0.97:
            # self.sig_stat.append( power1 )
            #sigma_mle = np.sqrt( np.sum(np.power(self.sig_stat, 2)) / (2 * len(self.sig_stat)) )
            # sigma_mle = np.sqrt( np.sum(self.sig_stat) / (2 * len(self.sig_stat)) )
            self.sigma_sig += 0.1 * (power1 - self.sigma_sig)
        # else:
            # self.noise_stat.append( power1 / self.sigma_nse )

        self.p_hist.append(p)
        # p_filt = p
        # p_filt = self.p_hist[-1] * np.sqrt(self.p_hist[-2])
        p_filt = np.sqrt(self.p_hist[-1] * self.p_hist[-2]) # * self.p_hist[-4]
        # p_filt = 2*(self.p_hist[-1] - 0.5) * (self.p_hist[-2] - 0.5) + 0.5
        self.p_stat.append( p_filt )
        self.D.process(p_filt)


class FSTDecoder:
    def __init__(self):
        self.F1 = self.create_f1()
        self.F2 = self.create_f2()
        self.F3 = self.create_f3()
        self.history = list()
        #initial_states = { 0: (0, None, '') }
        initial_states = { (0, 11, 0): (0, None, ('', '', '')) }
        self.history.append(initial_states)
        self.decoded = ''
        self.hypothesis = ''
        self.t = 0

    def process(self, p):
        if p < 1e-6:
            p = 1e-6
        if 1 - p < 1e-6:
            p = 1 - 1e-6
        self.t += 1
        states = self.history[-1]
        next_states = dict()
        for (sym_out, prob_sym) in (('0', np.log(1 - p)), ('1', np.log(p))):
            for state_from in states:
                prob_acc, _, _ = states[state_from]
                for (state_to, sym_in, prob_trans) in self.F1.translate_reverse(state_from[0], sym_out):
                    f2_match = self.F2.translate_reverse(state_from[1], sym_in)
                    # Allow epsilon:epsilon transitions
                    if sym_in == '' and len(f2_match) == 0:
                        f2_match = [ (state_from[1], '', 0) ]
                    for (state2_to, sym2_in, prob2_trans) in f2_match:
                        f3_match = self.F3.translate_reverse(state_from[2], sym2_in)
                        # Allow epsilon:epsilon transitions
                        if sym2_in == '' and len(f3_match) == 0:
                            f3_match = [ (state_from[2], '', 0) ]

                        for (state3_to, sym3_in, prob3_trans) in f3_match:
                            prob_new = prob_acc + prob_sym + prob_trans + prob2_trans + prob3_trans
                            state_new = (state_to, state2_to, state3_to)
                            if state_new not in next_states or prob_new > next_states[state_new][0]:
                                next_states[state_new] = (prob_new, state_from, (sym_in, sym2_in, sym3_in))

        idx = sorted(next_states, key=lambda x: -next_states[x][0])
        next_states = {key: next_states[key] for key in idx[:200]}
        # if idx[0] == (0, 11, 0):
        if idx[0] == (0, 0, 0):
        # if idx[0][1] == 0 and idx[0][2] == 0:
        # if idx[0][2] == 0:
            hypothesis = self.decode()
            if len(hypothesis) > len(self.hypothesis):
                self.hypothesis = hypothesis
                print(f'{self.decoded}{hypothesis}_')
        if idx[0] == (0, 11, 0):
            self.decoded += self.decode()
            self.hypothesis = ''
            self.history = list()
            # next_states = { (0, 11, 0): (0, None, ('', '', '')) }
            # pass
        self.history.append(next_states)

    def decode(self):
        states = self.history[-1]
        winner = max(states, key=lambda x: states[x][0])
        word_in = []
        for states in reversed(self.history):
            prob_acc, winner, sym_in = states[winner]
            word_in.append(sym_in)
        # print(''.join([x[0] for x in reversed(word_in)]))
        # print(''.join([x[1] for x in reversed(word_in)]))
        return (''.join([x[2] for x in reversed(word_in)]))

    def create_f1(self):
        F = FST()
        idx_state = 1

        def add_pdf(F, sym_in, sym_out, probs, idx_state):
            prob_sum = sum(probs)
            for length, prob in enumerate(probs):
                if prob == 0:
                    continue
                if length == 0:
                    F.add_arc(0, 0, sym_in, sym_out, np.log(prob / prob_sum))
                else:
                    q1 = 0
                    q2 = idx_state
                    while length > 0:
                        if q1 == 0:
                            F.add_arc(q1, q2, sym_in, sym_out, np.log(prob / prob_sum))
                        else:
                            F.add_arc(q1, q2, '', sym_out, 0)
                        idx_state += 1
                        q1 = q2
                        q2 = idx_state
                        length -= 1
                    F.add_arc(q1, 0, '', sym_out, 0)
            return idx_state

        idx_state = add_pdf(F, 'M', '1', [0, 1, 3, 3, 1], idx_state) # mark
        idx_state = add_pdf(F, '_', '0', [0, 1, 3, 3, 1], idx_state) # space
        F.add_final(0)
        return F

    def create_f2(self):
        F = FST()
        # Dot (M)
        F.add_arc(0, 1, '.', 'M', 0)
        # Dash (MMM)
        F.add_arc(0, 2, '-', 'M', 0)
        F.add_arc(2, 3, '', 'M', 0)
        F.add_arc(3, 1, '', 'M', 0)
        # Intersymbol space (_)
        F.add_arc(1, 0, '', '_', 0)
        # Intercharacter space (___)
        F.add_arc(1, 4, '|', '_', 0)
        F.add_arc(4, 5, '', '_', 0)
        F.add_arc(5, 0, '', '_', 0)
        # Interword space (_______ and longer)
        F.add_arc(1, 6, ' ', '_', 0)
        F.add_arc(6, 7, '', '_', 0)
        F.add_arc(7, 8, '', '_', 0)
        F.add_arc(8, 9, '', '_', 0)
        F.add_arc(9, 10, '', '_', 0)
        F.add_arc(10, 11, '', '_', 0)
        F.add_arc(11, 0, '', '_', 0)

        F.add_arc(11, 11, '', '_', 0)
        F.add_final(0)
        return F

    def create_f3(self):
        F = FST()
        idx_next = 2
        for sym_in in MORSE_TABLE:
            q1 = 0
            q2 = idx_next
            for sym_out in MORSE_TABLE[sym_in][:-1]:
                F.add_arc(q1, q2, '', sym_out, 0)
                idx_next += 1
                q1 = q2
                q2 = idx_next
            F.add_arc(q1, 1, sym_in, MORSE_TABLE[sym_in][-1], 0)

        F.add_arc(1, 0, '', '|', 0)
        F.add_arc(1, 0, ' ', ' ', 0)
        F.add_final(1)
        return F

class SlicerDecoder:
    def __init__(self, wpm=20, fps=65):
        t_unit = 1200 / wpm
        t_frame = 1000 / fps
        self.min_unit = 0.5 * t_unit / t_frame
        self.min_long = 2.0 * t_unit / t_frame
        self.min_word = 5.0 * t_unit / t_frame
        self.thr_lo = 0.35
        self.thr_hi = 0.65
        self.is_mark = False
        self.counter = [0, 0]
        self.elements = ''
        self.decoded = ''
        self.lookup = {elements: char for (char, elements) in MORSE_TABLE.items()}
        self.mark_mix = np.array([ [0.5, 3.0, 1.0], [0.5, 8.0, 2.0] ])
        pass

    def process(self, p):
        if self.is_mark:
            if p < self.thr_lo:
                self.is_mark = False
                if self.counter[1] > self.min_unit:
                    self.counter[0] = 1
                    self.process_mark(self.counter[1])
            else:
                self.counter[1] += 1
        else:
            if p > self.thr_hi:
                self.is_mark = True
                if self.counter[0] > self.min_unit:
                    self.counter[1] = 1
                    self.process_space(self.counter[0])
            else:
                self.counter[0] += 1

    def process_mark(self, count):
        p_k_sum = 0
        gamma = []
        for (pi, mu, sigma) in self.mark_mix:
            p_k = pi * pdf_normal(count, mu, sigma)
            gamma.append(p_k)
            p_k_sum += p_k
        gamma = np.array(gamma) / p_k_sum
        for k in range(len(self.mark_mix)):
            if gamma[k] < 1e-1:
                continue
            mu = self.mark_mix[k][1]
            pi_est = 1
            mu_est = count
            sigma_est = np.abs(count - mu)
            est = np.array((pi_est, mu_est, sigma_est))
            self.mark_mix[k] += 0.05 * (est - self.mark_mix[k])
        p_k_sum = 0
        for (pi, mu, sigma) in self.mark_mix:
            p_k_sum += pi
        for k in range(len(self.mark_mix)):
            self.mark_mix[k][0] /= p_k_sum
        # print(count, gamma, self.mark_mix[0][1], self.mark_mix[1][1], self.mark_mix[0][2], self.mark_mix[1][2])

        # if count < self.min_long:
        if gamma[0] > 0.5:
            self.elements += '.' # dit
        #elif count < self.min_word:
        elif gamma[1] > 0.5:
            self.elements += '-' # dash

    def process_space(self, count):
        if count < self.min_long:
            pass # interelement space
        elif count < self.min_word:
            self.elements += '|' # intercharacter space
        else:
            self.elements += ' ' # interword space

    def decode(self):
        print(f'Dit length={self.mark_mix[0][1]:.2f}, ratio={self.mark_mix[1][1]/self.mark_mix[0][1]:.1f}')
        print(self.elements)
        decoded = list()
        for word in self.elements.split(' '):
            if word == '':
                continue
            word_decoded = ''
            for char in word.split('|'):
                if char in self.lookup:
                    word_decoded += self.lookup[char]
                else:
                    word_decoded += '_'
                    pass
            decoded.append(word_decoded)
        return ' '.join(decoded)

def main():
    fs, sig = scipy.io.wavfile.read(sys.argv[1])
    sig = sig * (1/32768)
    fdet = float(sys.argv[2])
    wpm = float(sys.argv[3])

    t_unit = 1.2 / wpm
    samples_unit = fs * t_unit

    # N = 260                       # Analysis frame length
    S = int(samples_unit / 3.5)     # Analysis step size
    N = int(3.5*S)                  # Analysis frame length
    K = int(0.5 + (N * fdet / fs))  # Analysis frequency bin
    print(f'Sampling frequency: {fs}Hz')
    print(f'Detection frequency: {K*fs/N}Hz')

    M = CWMonitor(fs)
    D = FSTDecoder()
    # D = SlicerDecoder(fps=fs/S, wpm=wpm)
    frontend = CWFrontend(N, K, D)

    for idx in range(0, len(sig) - N, S):
        frontend.process(sig[idx:idx+N])
        # M.process(sig[idx:idx+N])

    D.decoded += D.decode()

    print(D.decoded)
    print()
    print(f'Sigma(signal) = {frontend.sigma_sig:.4f}')
    print(f'Sigma(noise) = {frontend.sigma_nse:.4f}')
    snr = 10 * np.log10(frontend.sigma_sig / frontend.sigma_nse)
    print(f'SNR = {snr:.1f} dB')

    # plt.hist(frontend.sig_stat_norm, bins=np.linspace(0, 4, 20), density=True)
    # plt.hist(-np.array(frontend.noise_stat), bins=np.linspace(-4, 0, 20), density=True)
    # x1 = np.linspace(0, 4, 100)
    # x2 = np.linspace(0.1, 4, 100)
    # y1 = [pdf_rayleigh(x_i) for x_i in x1]
    # y2 = [pdf_chi2_1dim(x_i) for x_i in x2]
    # plt.plot(x1, y1)
    # plt.plot(-x2, y2)
    # plt.plot(frontend.p_stat, marker='.')
    # plt.hist(-np.array(frontend.count_stats[0]), bins=range(-30, 15))
    # plt.hist(frontend.count_stats[1], bins=range(-30, 15))
    # plt.scatter(frontend.count_stats[0], frontend.count_stats[1])
    # plt.imshow(np.tile(frontend.p_stat, (200, 1)), vmin=0, vmax=1, cmap=plt.get_cmap('viridis'))
    # plt.grid()
    # plt.show()

if __name__ == '__main__':
    main()
    sys.exit(0)
