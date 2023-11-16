import math
import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy
import time
import csv

x = np.random.normal(size=100)
t = np.arange(len(x))
print("len(t):", t, "\n")

fig, ax = plt.subplots(4)
ax[0].plot(t, x)

for i in range (3):
    # conv = scipy.signal.fftconvolve(x, x)
    x = np.convolve(x, x)
    t = np.arange(len(x))
    print("len(t):", t, "\n")
    ax[i+1].plot(t, x)

plt.suptitle("Ex 1")
plt.tight_layout()
plt.show()

# Ex 2

def gen_polinoame(N):
    return np.random.randint(-20, 20, N)

p = np.polynomial.Polynomial(gen_polinoame(4))
q = np.polynomial.Polynomial(gen_polinoame(5))

r = np.polynomial.polynomial.polymul(p, q)

p1 = np.pad(p.convert().coef, (0, len(q.convert().coef) - 1))
q1 = np.pad(q.convert().coef, (0, len(p.convert().coef) - 1))

print(p1)
print(q1)

p_fft = np.fft.fft(p1)
q_fft = np.fft.fft(q1)

r_fft = p_fft * q_fft
r_ifft = np.fft.ifft(r_fft)

print("r direct: ", r)
print("r_fft ", r_ifft.real)

# print("R: ", r, "\n")
# print("R_FFT: ", r_fft, "\n")

