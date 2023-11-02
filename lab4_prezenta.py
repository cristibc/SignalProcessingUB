import math
import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time
import pickle

def fourier_semnal(x):
    N = len(x)
    n = np.arange(N)
    omega = n.reshape((N, 1))
    X = np.dot(np.exp(-2j * np.pi * n * omega / N), x)
    return X

def sinusoidala(amp, freq): 
    return amp * np.sin(2 * np.pi * freq * t)

N_samples = [128, 256, 512, 1024, 2048, 4096, 8192]

x = []
X = []
X2 = []
fourier_me = []
fourier_fast = []

if os.path.exists("fourier_me.npy") == 0 or os.path.exists("fourier_fast.npy") == 0:
    for i, sample in enumerate(N_samples):
        t = np.linspace(0, 1, sample, endpoint=False)
        x.append(sinusoidala(1, 1000))
        ms1 = int(round(time.time()*1000))
        X.append(fourier_semnal(x[i]))
        ms2 = int(round(time.time()*1000))
        delta_t = ms2 - ms1
        fourier_me.append(delta_t)
        
        ms1 = int(round(time.time()*1000))
        X2.append(np.fft.fft(x[i]))
        ms2 = int(round(time.time()*1000))
        delta_t = ms2 - ms1
        fourier_fast.append(delta_t)

        np.save('fourier_me', fourier_me)
        np.save('fourier_fast', fourier_fast)

fourier_me = np.load('fourier_me.npy')
fourier_fast = np.load('fourier_fast.npy')

plt.yscale("log")
plt.plot(N_samples, fourier_fast, color='b')
plt.plot(N_samples, fourier_me, color='r')
plt.xlabel("Sample size")
plt.ylabel("exec time in ms")
plt.show()

print(fourier_fast)
print(fourier_me)
