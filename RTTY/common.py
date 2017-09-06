import sys, wave, struct
import numpy as np
import scipy.signal as spsig
import scipy.fftpack as spfft

# Christopher P. Matthews
# christophermatthews1985@gmail.com
# Sacramento, CA, USA

def levenshtein(s, t):
        ''' From Wikipedia article; Iterative with two matrix rows. '''
        if s == t: return 0
        elif len(s) == 0: return len(t)
        elif len(t) == 0: return len(s)
        v0 = [None] * (len(t) + 1)
        v1 = [None] * (len(t) + 1)
        for i in range(len(v0)):
            v0[i] = i
        for i in range(len(s)):
            v1[0] = i + 1
            for j in range(len(t)):
                cost = 0 if s[i] == t[j] else 1
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            for j in range(len(v0)):
                v0[j] = v1[j]
                
        return v1[len(t)]

def read_wav(path):
    f = wave.open(path)
    fs = f.getframerate()
    nframes = f.getnframes()
    raw = f.readframes(nframes)
    x = struct.unpack('<' + 'h' * nframes, raw)
    return (np.array(x) / 32768.0, fs)

def write_wav(path, x, fs):
    f = wave.open(path, 'w')
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(fs)
    nframes = len(x)
    raw = np.array(x)
    if np.max(np.abs(raw)) > 1.0:
        raw /= np.max(np.abs(raw))
    raw = (raw * 32767.0).astype(int)
    f.writeframes(struct.pack('<' + 'h' * nframes, *raw))

def sinc(x):
    if np.fabs(x) < 1E-6: 
        return 1 - np.fabs(x)
    
    return np.sin(np.pi * x) / (np.pi * x)

def rcos_filter(beta, T, fs):
    N = int(fs * T * 2) * 2 + 1
    h = np.zeros(N)
    for i in range(N):
        t = (i - (N/2)) / float(fs)
        k0 = 2 * beta * t / T
        k1 = np.cos(np.pi * beta * t / T) / (1 - k0*k0)
        k2 = sinc(t / T)
        h[i] = k1 * k2 / T
    h = h / np.sum(h)
    return h

def rrc_filter(beta, T, fs):
    N = int(fs * T * 2) * 2 + 1
    h = np.zeros(N)
    for i in range(N):
        if i == N/2:
            h[i] = (1 + beta * (4/np.pi - 1)) / T
        else:
            t = (i - (N/2)) / float(fs)
            kt = t / T
            k1 = np.sin(np.pi * kt * (1 - beta))
            k2 = np.cos(np.pi * kt * (1 + beta))
            k3 = 4 * beta * kt
            k4 = (k1 + k3 * k2) / (np.pi * kt * (1 - k3 * k3))
            h[i] = k4 / T
    h = h / np.sum(h)
    return h
