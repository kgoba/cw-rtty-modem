#!/usr/bin/env python

import sys
import math
import random
import numpy as np


MORSE_TABLE = {
    'E':    '.',
    'T':    '-',
    'I':    '..',
    'A':    '.-',   
    'N':    '-.',
    'M':    '--',
    'S':    '...',
    'U':    '..-',
    'R':    '.-.',
    'W':    '.--',
    'D':    '-..',
    'K':    '-.-',
    'G':    '--.',
    'O':    '---',
    'H':    '....',
    'V':    '...-',
    'F':    '..-.',
#   ' ':    '..--',    
    'L':    '.-..',
#   ' ':    '.-.-',
    'P':    '.--.',
    'J':    '.---',    
    'B':    '-...',
    'X':    '-..-',
    'C':    '-.-.',
    'Y':    '-.--',   
    'Z':    '--..',
    'Q':    '--.-',
#   ' ':    '---.',
#   ' ':    '----',
    '0':    '-----',
    '1':    '.----',
    '2':    '..---',
    '3':    '...--',
    '4':    '....-',
    '5':    '.....',
    '6':    '-....',
    '7':    '--...',
    '8':    '---..',
    '9':    '----.',
    '<KN>': '-.--.', # also as (
    '+': '.-.-.', # also as <AR>
    '&': '.-...', # also as <AS>
    '<CT>': '-.-.-',
    '=':    '-...-', # also <BT>
    '/':    '-..-.',
    '?':    '..--..',
    ',':    '--..--',
    '.':    '.-.-.-',
    '<SK>': '...-.-'
}


def stringToKey(str):
    key = ''
    lastChar = False
    for c in str:
        thisChar = False
        uc = c.upper()
        if c == '\n':
            key += c
        elif c == ' ':
            key += c
        elif uc in MORSE_TABLE:
            thisChar = True
            if lastChar:
                key += '`'
            key += MORSE_TABLE[uc]
        lastChar = thisChar
    return '`' + key  + '`'

def keyToTimings(key, wpm = 25):
    s = wpm
    c = wpm
    if wpm < 18:
        c = 18
    u = 1.2 / c     # e.g. 48 ms for 25 WPM
    ta = (60 * c - 37.2 * s) / (s * c)
    tc = 3 * ta / 19    # intercharacter period
    tw = 7 * ta / 19    # interword period

    timings = []
    lastOn = False
    for c in key:
        thisOn = False
        if c == '\n':
            timings.append(-3 * tw)
        elif c == ' ':
            timings.append(-tw)
        elif c == '`':
            # timings.append(-tc)
            timings.append(-u)
        elif c == '.':
            thisOn = True
            if lastOn:
                timings.append(-u)
            timings.append(u)
        elif c == '-':
            thisOn = True
            if lastOn:
                timings.append(-u)
            timings.append(3*u)
        lastOn = thisOn
    return timings

def timingsToEnvelope(timings, fs = 200):
    env = []
    dt = 1.0 / fs
    target = 0
    for t in timings:
        target += math.fabs(t)
        while (dt * len(env)) < target:
            if t > 0:
                env.append(1.0)
            else:
                env.append(-1.0)
                
    return env

def generateRandomString():
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    words = []
    for idxWord in range(10):
        length = int(0.5 + random.gauss(4.5, 1.5))
        word = ''.join([random.choice(chars) for x in range(length)])
        words.append(word)
    return ' '.join(words)

def generateTrainingSample(wpm = 20, fs = 50):
    string = generateRandomString()
    key = stringToKey(string)
    timings = keyToTimings(key, wpm)
    env = timingsToEnvelope(timings, fs)
    
    return (string, env)

def generateCorrelators(wpm = 20, fs = 50):
    cor = dict()
    for c in MORSE_TABLE:
        key = stringToKey(c)
        timings = keyToTimings(key, wpm)
        env = timingsToEnvelope(timings, fs)
        cor[c] = np.array(env)
        cor[c] -= np.mean(cor[c])
    print(cor['E'])
    print(cor['T'])
    return cor

def example1():
    string = 'ALL YOUR BASE ARE BELONG US'
    key = stringToKey(string)
    timings = keyToTimings(key, 20)
    env = timingsToEnvelope(timings, 50)

    print(string)
    print(key)
    print(timings)
    print(env)

def extractEnv(sig, fsOrig, fs):
    #bandpass = spsig.firwin(50, [f1, f2], pass_zero=False)
    env = np.abs(sig)
    lowpass = spsig.firwin(50, 100.0, nyq = fsOrig / 2.0)
    env = spsig.lfilter(lowpass, 1, env)

    for N in range(1, 50):
        M = fsOrig * N / float(fs)
        M = int(0.5 + M)
        fs2 = float(fsOrig * N) / M
        relErr = math.fabs(fs2/fs - 1.0)
        if relErr < 1E-3:
            print(N, "/", M)
            break

    env = spsig.resample_poly(env, N, M)
    #env -= np.mean(env)
    env /= np.max(env)
    
    return env

if __name__ == '__main__':
    import wave
    import struct
    import scipy.signal as spsig

    wav = wave.open(sys.argv[1])
    fs = wav.getframerate()
    assert(wav.getnchannels() == 1)
    assert(wav.getsampwidth() == 2)
    count = wav.getnframes()
    sig = struct.unpack('<' + 'h' * count, wav.readframes(count))
    sig = np.array(sig) / 32768.0

    fs2 = 125
    wpm = 22

    env = extractEnv(sig, fs, fs2)
    corr = generateCorrelators(wpm = wpm, fs = fs2)

    c1 = spsig.convolve(env, corr['E'], mode = 'same')
    c2 = spsig.convolve(env, corr['T'], mode = 'same')

    import matplotlib.pyplot as plt

    #plt.plot(env[:1000])
    plt.plot(c1[:1000])
    plt.plot(c2[:1000])
    # plt.plot(c1[:1000], c2[:1000])
    plt.grid()
    plt.show()
    #print(c1[200:280])
    #print(c2[200:280])

    sys.exit(0)
