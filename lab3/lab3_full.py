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
    ax[i].plot(np.real(n), color='#1ca6ff')
    ax[i].plot(np.imag(n), color='#ff1ca6')
    ax[i].legend(["real", "imag"], loc ="lower right") 
    i += 1

plt.xlabel("k")
plt.ylabel("m")
plt.savefig('ex1.png')
plt.savefig('ex1.pdf')
plt.show()

# Ex 2

eroare = np.linalg.norm(np.abs(np.dot(np.conjugate(fourier_test.T), fourier_test)) - len(fourier_test) * np.identity(fourier_test.shape[0]), 2)
if eroare < 10**-10:
    print("Matricea este unitara")
else: 
    print("Matricea nu este unitara")
    
samples = 1000

t = np.linspace(0, 1, samples, endpoint=False)
def sinusoidala(amp, freq): 
    return amp * np.sin(2 * np.pi * freq * t)

x = sinusoidala(1, 15)
x_infasurat = x * np.exp(-2j * np.pi * t)

plt.figure(figsize = (11, 4))
plt.subplot(1, 2, 1)
plt.xlabel("Timp(esantioane)")
plt.ylabel("Amplitudine")
plt.plot(t, x, color='#ff4e1c')
plt.subplot(1, 2, 2)
plt.xlabel("Real")
plt.ylabel("Imaginar")
plt.plot(np.real(x_infasurat), np.imag(x_infasurat), color='#c01cff')
plt.suptitle('Exercitiul 2 - Figura 1')
plt.savefig('ex2.1.png')
plt.savefig('ex2.1.pdf')
plt.show()

w = [1, 3, 6, 15]

plt.figure(figsize = (7,6))
for i, w in enumerate(w):
    z = x * np.exp(-2j * np.pi * t * w)

    plt.subplot(2, 2, i+1)
    plt.plot(np.real(z), np.imag(z), color='#29d100')
    plt.title(f'ω = {w}')
    plt.xlabel("Real")
    plt.ylabel("Imaginar")

plt.tight_layout()
plt.suptitle('Exercitiul 2 - Figura 2')
plt.savefig('ex2.2.png')
plt.savefig('ex2.2.pdf')
plt.show()

# Ex 3

x1 = sinusoidala(0.5, 15)
x2 = sinusoidala(1, 20)
x3 = sinusoidala(2, 75)

semnal_final = x1 + x2 + x3

plt.figure(figsize = (11, 5))
plt.subplot(1, 2, 1)
plt.xlim(0, 0.2)
plt.grid(True)
plt.xlabel("Timp (s)")
plt.ylabel("x(t)")
plt.plot(t, semnal_final, color='#6600ff')

def fourier_semnal(x):
    N = len(x)
    n = np.arange(N)
    omega = n.reshape((N, 1))
    X = np.dot(np.exp(-2j * np.pi * n * omega / N), x)
    return X

frecventa = np.arange(len(semnal_final))

plt.subplot(1, 2, 2)
plt.stem(frecventa, np.abs(fourier_semnal(semnal_final)), linefmt = '#00b6d8')
plt.xlim(0, 100)
plt.suptitle('Exercitiul 3')
plt.grid(True)
plt.xlabel("Frecventa (Hz)")
plt.ylabel("|X(ω)|")
plt.savefig('ex3.png')
plt.savefig('ex3.pdf')
plt.show()