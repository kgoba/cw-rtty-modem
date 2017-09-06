import sys, wave, struct, math
import numpy as np
import scipy.signal as spsig
import scipy.fftpack as spfft
from Queue import Queue

from common import *


class BaudotEncoder:
    LETTERS = ' T\rO HNM' + '\nLRGIPCV' + 'EZDBSYFX' + 'AWJ UQK '
    # US version of the figures case
    FIGURES_US = ' 5\r9 #,.' + '\n)4&80:;' + '3"$? 6!/' + '-2\' 71( '
    FIGURE_SHIFT = 0x1B
    LETTER_SHIFT = 0x1F

    def __init__(self):
        self.isFigureMode = False
        self.letter_map = dict([(c, idx) for (idx, c) in enumerate(self.LETTERS) if c != ' '])
        self.letter_map[' '] = 0x04
        self.figure_map = dict([(c, idx) for (idx, c) in enumerate(self.FIGURES_US) if c != ' '])
        self.figure_map[' '] = 0x04
        self.output = Queue()
        pass
    
    def write(self, character):
        if self.isFigureMode:
            if character in self.figure_map:
                self.output.put(self.figure_map[character])
            elif character in self.letter_map:
                self.isFigureMode = False
                self.output.put(self.LETTER_SHIFT)
                self.output.put(self.letter_map[character])
            else:
                return
        else:
            if character in self.letter_map:
                self.output.put(self.letter_map[character])
            elif character in self.figure_map:
                self.isFigureMode = True
                self.output.put(self.FIGURE_SHIFT)
                self.output.put(self.figure_map[character])
            else:
                return

    def available(self):
        return not self.output.empty()
        
    def read(self):
        return self.output.get()

class FrameEncoder:
    def __init__(self, bits = 5, rate = 50, fs = 8000, stopbit = 1.5):
        self.bits = 5
        self.stopbit = stopbit
        self.t_sym = 1.0 / rate
        self.dt = 1.0 / fs
        self.t = 0
        self.output = Queue()

    def _add_bit(self, bit, duration = 1.0):
        t2 = self.t + (self.t_sym * duration)
        while self.t < t2:
            self.output.put(bit)
            self.t += self.dt
    
    def write(self, data):
        mask = 1 << (self.bits - 1)
        self._add_bit(-1)
        for idx in range(self.bits):
            if data & mask: self._add_bit(1)
            else: self._add_bit(-1)
            mask >>= 1
        self._add_bit(1, self.stopbit)

    def available(self):
        return not self.output.empty()

    def read(self):
        return self.output.get()

class FSKEncoder:
    def __init__(self, fc = 800, shift = 450, fs = 8000):
        self.f1 = fc - shift / 2
        self.f2 = fc + shift / 2
        self.dt = 1.0 / fs
        self.phase = 0
        self.output = Queue()
   
    def write(self, data):
        if data > 0:
            self.phase += 2 * math.pi * self.dt * self.f1
        else:
            self.phase += 2 * math.pi * self.dt * self.f2
        self.output.put(math.sin(self.phase))

    def available(self):
        return not self.output.empty()

    def read(self):
        return self.output.get()

fs = 8000
#text = "QWE 123 ABC 987"
text = open(sys.argv[1]).read()
encoder1 = BaudotEncoder()
encoder2 = FrameEncoder(fs = fs)
encoder3 = FSKEncoder(fs = fs)

for idx in range(fs / 10):
    encoder3.write(1)

for c in text:
    encoder1.write(c)

while encoder1.available():
    c2 = encoder1.read()
    print "0x%02x" % c2
    encoder2.write(c2)

while encoder2.available():
    c3 = encoder2.read()
    encoder3.write(c3)
    #print "%d" % c3

for idx in range(fs / 10):
    encoder3.write(1)

x = []
while encoder3.available():
    c4 = encoder3.read()
    x.append(c4)
    #print "%.2f" % c4
    
write_wav('out.wav', x, fs)
