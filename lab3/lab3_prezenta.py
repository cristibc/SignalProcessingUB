import math
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time
from scipy.io import wavfile

# Ex 1

X = []
def fourier(N):
    X = np.zeros((N, N), dtype=complex)
    for m in range (N):
        for k in range(N):
            X[m, k] = np.exp(-2j * np.pi * m * k / N)
    return X

fourier_test = fourier(8)
fig, ax = plt.subplots(8)
fig.suptitle('Exercitiul 1')
i=0
for n in fourier_test:
    ax[i]
    ax[i].plot(np.real(n))
    ax[i].plot(np.imag(n))
    ax[i].legend(["real", "imag"], loc ="lower right") 
    i += 1

plt.xlabel("k")
plt.ylabel("m")
plt.show()

eroare = np.linalg.norm(np.abs(np.dot(np.conjugate(fourier_test.T), fourier_test)) - len(fourier_test) * np.identity(fourier_test.shape[0]), 2)
if eroare < 10**-10:
    print("Matricea este unitara")
else: 
    print("Matricea nu este unitara")

def infasurare(x):
    y = np.zeros((len(x), len(x)), dtype=complex)
    for n in range(len(x)):
        y[n] = x[n] * np.exp(-2j * np.pi * n)
        return y
    
samples = 1000

t = np.arange(0, 1, 1/samples)
def sinusoidala(freq): 
    return np.sin(2 * np.pi * freq * t)

x_infasurat = infasurare(sinusoidala(100))

plt.plot(np.real(x_infasurat), np.imag(x_infasurat))
plt.show()