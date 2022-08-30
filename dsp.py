import numpy as np

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
    # no normalization!
    return power
