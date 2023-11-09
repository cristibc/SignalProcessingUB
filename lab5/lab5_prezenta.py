import math
import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time
import csv

# Ex 1

# a)

x = np.genfromtxt('lab5/Train.csv', delimiter=',', skip_header=1)
print(x)
X = np.fft.fft(x)
print(X.shape)
print(X)

N = len(x)
N_halved = int(N/2)
X = abs(X/N)
print(X)
X = X[:N_halved]

# Avem un sample pe ora, din formula Fs = NrSamples / UnitateTimp
# deci impartim nr de samples la 24 ca sa aflam cate zile au fost sample-uite
Fs = N / (N/24)
f = Fs * np.linspace(0, N_halved, N_halved)/N

# b)
print("Intervalul de timp este de: ", int(N/24), " zile")

# c) 

fmax = 0.5 * Fs
print("Frecventa maxima: ", fmax)
t = np.arange(N_halved)

plt.plot(f, X)
plt.yscale("log")
plt.show()