import math
import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from time import perf_counter_ns
import librosa

# Ex 1

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
        ms1 = perf_counter_ns()
        X.append(fourier_semnal(x[i]))
        ms2 = perf_counter_ns()
        delta_t = ms2 - ms1
        fourier_me.append(delta_t)
        
        ms1 = perf_counter_ns() 
        X2.append(np.fft.fft(x[i]))
        ms2 = perf_counter_ns()
        delta_t = ms2 - ms1
        fourier_fast.append(delta_t)

        np.save('fourier_me', fourier_me)
        np.save('fourier_fast', fourier_fast)

fourier_me = np.load('fourier_me.npy')
fourier_fast = np.load('fourier_fast.npy')

plt.yscale("log")
plt.plot(N_samples, fourier_me, color='b')
plt.plot(N_samples, fourier_fast, color='r')
plt.xlabel("Sample size")
plt.ylabel("exec time in ns")
plt.legend(["slowFourier", "fastFourier"], loc ="lower right")
plt.suptitle('Exercitiul 1')
plt.grid(True)
plt.savefig('ex1.png')
plt.savefig('ex1.pdf')
plt.show()

# Ex 2

def sinusoidala(freq, t): 
    return np.sin(2 * np.pi * freq * t)

f0 = 25
fs = 40

t_original = np.linspace(0, 1, 1000, endpoint=False)
x_original = sinusoidala(fs, t_original)

f1 = f0
f2 = f0 + fs * 1
f3 = f0 + fs * 2

t = np.linspace(0, 1, fs, endpoint=False)

freqs = [f0, f0 + fs * 1, f0 + fs * 2]
x = []
x_resampled = []

fig, ax = plt.subplots(4)

ax[0].plot(t_original, sinusoidala(f0, t_original))
ax[0].set_xlim([0, 0.2])

for i, freq in enumerate(freqs):
    x.append(sinusoidala(freq, t))
    x_resampled.append(sinusoidala(freq, t_original))

    ax[i+1].plot(t_original, x_resampled[i], color='#969696')
    ax[i+1].plot(t_original, x_resampled[i], color='#ff9742')
    ax[i+1].scatter(t, x[i], color='#ff8000')
    ax[i+1].set_xlim([0, 0.2])

plt.suptitle('Exercitiul 2')
plt.tight_layout()
plt.savefig('ex2.png')
plt.savefig('ex2.pdf')
plt.show()

# 3

f0 = 25
fs = 100

t_original = np.linspace(0, 1, 1000, endpoint=False)
x_original = sinusoidala(fs, t_original)

f1 = f0
f2 = f0 + fs * 1
f3 = f0 + fs * 2

t = np.linspace(0, 1, fs, endpoint=False)

freqs = [f0, f0 + fs * 1, f0 + fs * 2]
x = []
x_resampled = []

fig, ax = plt.subplots(4)

ax[0].plot(t_original, sinusoidala(f0, t_original))
ax[0].set_xlim([0, 0.2])

for i, freq in enumerate(freqs):
    x.append(sinusoidala(freq, t))
    x_resampled.append(sinusoidala(freq, t_original))

    ax[i+1].plot(t_original, x_resampled[i], color='#969696')
    ax[i+1].plot(t_original, x_resampled[i], color='#ff9742')
    ax[i+1].scatter(t, x[i], color='#ff8000')
    ax[i+1].set_xlim([0, 0.2])

plt.suptitle('Exercitiul 3')
plt.tight_layout()
plt.savefig('ex3.png')
plt.savefig('ex3.pdf')
plt.show()

# Ex 4

freq_min = 2 * max(40, 200)
print("Frecventa minima este: ", freq_min, "Hz")

# Ex 5

# Sunt destul de similare, am atasat in vocale.png

# Ex 6

y, sr = librosa.load("sound.wav", sr=None)

grup_size = int(0.01 * len(y))
overlap = int(0.5 * grup_size)

grupuri = []

for i in range(0, len(y) - grup_size, overlap):
    grup = y[i : i + grup_size]
    grupuri.append(np.abs(np.fft.fft(grup)))

semnal_grupat = np.array(grupuri).T
print(semnal_grupat)

librosa.display.specshow(librosa.amplitude_to_db(semnal_grupat, ref=np.max),
                         y_axis='log', x_axis='time')
plt.colorbar(format='%+2.f dB')
plt.suptitle('Exercitiul 6')
plt.xlabel('timp')
plt.ylabel('Hz')
plt.savefig('ex6.png')
plt.savefig('ex6.pdf')
plt.show()

# Ex 7

# SNRdB = 10 * log(Ps / Pnoise, 10)
Ps = 90
SNR_db = 80
P_noise = Ps / 10 ** (SNR_db/10)
print(P_noise)