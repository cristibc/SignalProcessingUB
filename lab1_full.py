import numpy as np
from matplotlib import pyplot as plt
import random
import sounddevice as sd
import time
from scipy.io import wavfile

t = np.linspace(0, 0.03, 200)
def xfun(t):
    return np.cos(520 * np.pi * t + np.pi / 3)
def yfun(t):
    return np.cos(280 * np.pi * t - np.pi / 3)
def zfun(t):
    return np.cos(120 * np.pi * t + np.pi / 3)

t1 = np.linspace(0, 0.003, 6)

fig, ax = plt.subplots(3)
fig.suptitle('Exercitiul 1')

for i in range (0,3):
    ax[i].axhline(y=0, color='k')
ax[0].plot(t, xfun(t))
ax[0].stem(t1, xfun(t1))
ax[0].set(xlabel='t', ylabel='x')
ax[1].plot(t, yfun(t))
ax[1].stem(t1, yfun(t1))
ax[1].set(xlabel='t', ylabel='y')
ax[2].plot(t, zfun(t))
ax[2].stem(t1, zfun(t1))
ax[2].set(xlabel='t', ylabel='z')
ax[0].axhline(y=0, color='k')
ax[0].axvline(x=0, color='k')
for ax in ax.flat:
    ax.set_xlim([0, 0.003])
plt.show()

sample1 = 1600
t1 = np.linspace(0, 1, sample1, endpoint=False)

def sine1(hz, time):
    return np.sin(2 * np.pi * hz * time)

fig, ax = plt.subplots(4)
fig.suptitle('Exercitiul 2 (a,b,c,d)')

ax[0].plot(t1, sine1(400, t1))
ax[0].set_xlim([0, 0.02])
ax[0].axhline(y=0, color='k')

sample2 = 16000 * 3
t2 = np.linspace(0, 3, sample2, endpoint=False)

def sine2(hz, time):
    return np.sin(2 * np.pi * hz * time)

ax[1].plot(t2, sine2(800, t2))
ax[1].set_xlim([0, 0.02])
ax[1].axhline(y=0, color='k')

t3 = np.linspace(0, 1, 48000, endpoint=False)

def sawtooth(hz, time):
    return 2 * (time * hz - np.floor(time * hz + 0.5))

ax[2].plot(t3, sawtooth(240, t3))
ax[2].set_xlim([0, 0.02])
ax[2].axhline(y=0, color='k')

t4 = np.linspace(0, 1, 60000, endpoint=False)

def square(hz, time):
    return np.sign(np.sin(2* np.pi * hz * time))

ax[3].plot(t4, square(300, t4))
ax[3].set_xlim([0, 0.02])
ax[3].axhline(y=0, color='k')
plt.show()

sd.play(sine1(400, t1) * 0.01, 1600)
time.sleep(1)
sd.play(sine2(800, t2) * 0.01, 1600)
time.sleep(1)
sd.play(sawtooth(240, t3) * 0.002, 48000)
time.sleep(1)
sd.play(square(300, t4) * 0.002, 60000)
time.sleep(1)
sd.stop()

wavfile.write("sine1.wav", 100000, (sine1(400, t1) * 0.01).astype(np.float32))
wavfile.write("sine2.wav", 100000, (sine2(800, t2) * 0.01).astype(np.float32))
wavfile.write("sawtooth.wav", 100000, (sawtooth(240, t3) * 0.01).astype(np.float32))
wavfile.write("square.wav", 100000, (square(300, t4) * 0.01).astype(np.float32))

randomSignal = np.random.rand(128, 128)

fig, ax = plt.subplots(1, 2)
fig.suptitle('Exercitiul 2 (e,f)')
ax[0].imshow(randomSignal)

custom2d = np.zeros((128, 128))
for i in range(len(custom2d)):
    for j in range (len(custom2d[i])):
        if (i * len(custom2d[i]) + j) % 7 == 0:
            custom2d[i][j] = random.uniform(0, 1)
        else:
            custom2d[i][j] = random.uniform(2, 3)
        
ax[1].imshow(custom2d)
plt.colorbar
plt.show()

sampling_rate = 2000
time = 1 / sampling_rate

time_sampled = 3600
bits_per_sample = 4
nr_samples = sampling_rate * time_sampled
total_bytes = nr_samples * (bits_per_sample/8)

print("Time between samples: ", time)
print("Total bytes: ", total_bytes)