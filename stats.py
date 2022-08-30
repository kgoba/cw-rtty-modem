import numpy as np

def pdf_chi2_1dim(x, scale=1):
    ''' Probability distribution function of scaled chi-squared distribution with nu=1 dimension,
        i.e. distribution of power if the amplitude is normal distributed.
        Equals Gamma(k=1/2, theta=2*scale).
    '''
    y = x / scale
    return np.exp(-y / 2) / np.sqrt(2 * np.pi * y)

def pdf_half_normal(x, sigma=1):
    return np.sqrt(2 / np.pi) / sigma * np.exp(-x*x / (2*sigma*sigma))

def pdf_chi2_2dim(x, scale=1):
    ''' Probability distribution function of scaled chi-squared distribution with nu=2 dimensions,
        i.e. distribution of power of a complex amplitude with normal distributed real and imaginary parts (iid).
        Equals Gamma(k=1, theta=2*scale) ~ Exponential(rate=1/(2*scale)).
    '''
    y = x / scale
    return np.exp(-y / 2) / 2

def pdf_rayleigh(x, sigma=1):
    return x / (sigma * sigma) * np.exp(-x*x / (2*sigma*sigma))

def pdf_normal(x, mu, sigma):
    x_norm = (x - mu) / sigma
    return np.exp( -(x_norm**2)/2 ) / sigma / np.sqrt(2*np.pi)
