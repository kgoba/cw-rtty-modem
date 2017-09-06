import sys, wave, struct
import numpy as np
import scipy.signal as spsig
import scipy.fftpack as spfft
from Queue import Queue

from common import *

class DumbTerminal:
    def __init__(self):
        pass
    
    def write(self, char):
        if len(char) > 1:
            return
        if char == '\r':
            #sys.stderr.write('\n')
            return
        sys.stderr.write(char)

class BaudotDecoder:
    LETTERS = ' T\rO HNM' + '\nLRGIPCV' + 'EZDBSYFX' + 'AWJ UQK '
    # US version of the figures case
    FIGURES_US = ' 5\r9 #,.' + '\n)4&80:;' + '3"$? 6!/' + '-2\' 71( '
    FIGURE_SHIFT = 0x1B
    LETTER_SHIFT = 0x1F

    def __init__(self):
        self.isFigureMode = False
        self.receiver = DumbTerminal()
        self.output = Queue()
        pass
    
    def write(self, frame):
        if self.isFigureMode:
            if frame == self.LETTER_SHIFT:
                self.isFigureMode = False
            else:
                self.output.put(self.FIGURES_US[frame & 0x1F])
        else:
            if frame == self.FIGURE_SHIFT:
                self.isFigureMode = True
            else:
                self.output.put(self.LETTERS[frame & 0x1F])

    def available(self):
        return not self.output.empty()
        
    def read(self):
        return self.output.get()

class FrameExtractor:
    def __init__(self, fs, rate):
        self.fs     = fs
        self.rate   = rate
        self.samples_per_symbol = fs / rate
        self.n_bits = 5
        #print "samples per symbol: %d" % self.samples_per_symbol
        
        self.window = [1] * int(0.5 + self.samples_per_symbol)
        
        self.count = 0
        self.n_mark = len(self.window)
        self.n_space = 0
        self.state = 0
        self.frame = 0
        self.output = Queue()
        pass
    
    def detect_start(self):
        N = len(self.window)
        half = int(N/2)
        if self.window[half] < 0 or self.window[half + 1] >= 0:
            return False
        err = 0
        for idx in range(N):
            if idx <= half and self.window[idx] < 0: err += 1
            if idx > half and self.window[idx] >= 0: err += 1
        return err <= int(N/4)
    
    def detect_mark(self):
        N = len(self.window)
        half = int(N/2)
        return self.n_space < half
        #return self.window[half] > 0           

        err = 0
        for idx in range(0, N):
            if self.window[idx] < 0:
                err += 1
        return err < half
        #for idx in range(half - 3, half + 4):
        #    if self.window[idx] < 0:
        #        err += 1
        #return err <= 3

    def shift(self, x):
        N = len(self.window)
        
        if self.window[0] < 0: self.n_space -= 1
        else: self.n_mark -= 1
            
        for idx in range(1, N):
            self.window[idx - 1] = self.window[idx]
        self.window[N-1] = x

        if self.window[N-1] < 0: self.n_space += 1
        else: self.n_mark += 1

        return
        
    def write(self, x):
        # update symbol window
        self.shift(x)

        N = len(self.window)

        if self.count > 0:
            self.count -= 1
        else:
            if self.state == 0:
                if self.detect_start():
                    self.state = 1
                    self.frame = 0
                    self.count = int(N/2)
            else:
                if self.state > self.n_bits + 1:
                    self.state = 0
                    # all bits received, check for the stop bit
                    if self.detect_mark():
                        self.output.put(self.frame)
                else:
                    if self.state == 1 and self.detect_mark():
                        self.state = 0
                    else:
                        self.state += 1
                        # update frame
                        self.frame <<= 1
                        if self.detect_mark():
                            self.frame |= 1
                        self.count = N

    def available(self):
        return not self.output.empty()
        
    def read(self):
        return self.output.get()

def plot():
    import matplotlib.pyplot as plt

    for idx in range(1):
        plt.plot(1000*t2[:712], -2 * idx + x7[712*idx:712*(idx + 1)])
        plt.plot(1000*t2[:712], -2 * idx + x8[712*idx:712*(idx + 1)])


    #H = np.abs(spfft.fft(x1[:20000].real))
    #H_f = np.linspace(0, fs, len(H))

    #plt.plot(H_f, 20 * np.log(H))
    #plt.xlim([0, fs/2])
    #plt.ylim([-40, 100])

    plt.grid()
    plt.show()

def rtty_demod(x, fs, rtty_fc, rtty_shift, rtty_rate):
    K = 5

    n = np.arange(len(x))
    t = n / float(fs)
    t2 = t * K
    fs2 = fs / float(K)

    het_cen = np.exp(-1j * 2*np.pi*rtty_fc/fs * n)
    het_dev = np.exp(1j * 2*np.pi*rtty_shift/2/fs * n)

    b = spsig.firwin(71, rtty_shift/8, nyq=fs/2, window='blackman')
    a = 1
    #(b, a) = spsig.butter(3, (rtty_shift/4) / (fs/2))
    #print 5000 * b, a
    #b2 = spsig.firwin(15, rtty_rate, nyq=fs2/2)
    #b2 = rrc_filter(0.51, 0.5/rtty_rate, fs2)
    b2 = rcos_filter(0.31, 0.5/rtty_rate, fs2)
    #print len(b2), b2

    x1 = spsig.lfilter(b, a, x * het_cen * het_dev)
    x2 = spsig.lfilter(b, a, x * het_cen * np.conj(het_dev))

    #x3 = x1 * np.conj(het_dev) + x2 * het_dev
    #x4 = np.unwrap(np.angle(x3))
    #x5 = -np.diff(x4) / (2*np.pi * rtty_shift/2/fs)   
    x5 = np.abs(x1) - np.abs(x2)

    x6 = spsig.decimate(x5, K, zero_phase = False)
    x7 = spsig.lfilter(b2, 1, x6)

    #x8 = np.zeros_like(x7)
    #last = -1
    #for idx in range(len(x7)):
    #    if x7[idx] < -0.7 and last > 0: last = -1
    #    elif x7[idx] > 0.7 and last < 0: last = 1
    #    x8[idx] = last

    block1 = FrameExtractor(fs2, rtty_rate)
    block2 = BaudotDecoder()
    block3 = DumbTerminal()
 
    for x in x7:
        block1.write(x)
    
    while block1.available():
        x = block1.read()
        block2.write(x)
    
    result = ''
    while block2.available():
        x = block2.read()
        result += x
        #block3.write(x)
    
    return result

def do_test(x, fs, text_orig):
    sig_db = 10 * np.log10(np.mean(x * x))
    for snr_db in np.linspace(0, -9, 4):
        noise_db = sig_db - snr_db
        dist = []
        for idx in range(5):
            sigma = np.power(10.0, noise_db / 20.0)
            noise = sigma * np.random.randn(len(x))
            text = rtty_demod(x + noise, fs, rtty_fc, rtty_shift, rtty_rate)
            dist.append(levenshtein(text, text_orig))
            write_wav('out_.wav', x + noise, fs)
            #print text
        print "SNR dB:", snr_db, "Distance:", np.mean(dist)

(x, fs) = read_wav(sys.argv[1])
x /= np.max(np.abs(x))

rtty_fc = 800.0
rtty_shift = 450.0
rtty_rate = 50.0

if len(sys.argv) > 2:
    text_orig = open(sys.argv[2]).read()
    do_test(x, fs, text_orig)

else:
    text = rtty_demod(x, fs, rtty_fc, rtty_shift, rtty_rate)
    print text
