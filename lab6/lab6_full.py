import math
import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import random
from scipy.signal import butter, cheby1, filtfilt, freqz

x = np.random.normal(size=100)
t = np.arange(len(x))

fig, ax = plt.subplots(4)
ax[0].plot(t, x)

for i in range (3):
    x = np.convolve(x, x)
    t = np.arange(len(x))
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


# Ex 3

def w_drept(N):
    return np.ones(N)

def w_hann(N):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / N))

np.hanning
N = 200
f = 100
A = 1

t = np.linspace(0, 1, N, endpoint=False)
semnal = A * np.sin(2 * np.pi * f * t)

fereastra_drept = w_drept(N)
fereastra_hann = w_hann(N)

semnal_drept = semnal * fereastra_drept
semnal_hann = semnal * fereastra_hann

plt.subplot(3, 2, 1)
plt.plot(t, fereastra_drept)
plt.title("Fereastra Dreptunghiulara")
plt.xlabel("Hz")
plt.ylabel("A")

plt.subplot(3, 2, 2)
plt.plot(t, semnal_drept)
plt.title("Semnal Dreptunghiular")
plt.xlabel("Hz")
plt.ylabel("A")

plt.subplot(3, 2, 3)
plt.plot(t, fereastra_hann)
plt.title("Fereastra Hann")
plt.xlabel("Hz")
plt.ylabel("A")

plt.subplot(3, 2, 4)
plt.plot(t, semnal_hann)
plt.title("Semnal Hann")
plt.xlabel("Hz")
plt.ylabel("A")

plt.subplot(3, 2, 5)
plt.plot(t, semnal)
plt.title("Semnal Original")
plt.xlabel("Hz")
plt.ylabel("A")

plt.tight_layout()
plt.show()

# Ex 4

# a)

x = np.genfromtxt('lab6/Train.csv', delimiter=',', skip_header=1, usecols=2)
x_sliced = x[9576:9648]

# b)
t = np.linspace(0, 1, len(x_sliced), endpoint=False)

w_values = [4, 10, 16, 25]
x_filtered = []

plt.plot(t, x_sliced)

for i, w in enumerate(w_values):
    x_filtered.append(np.convolve(x_sliced, np.ones(w), 'valid') / w)
    t_new = np.linspace(0, 1, num=len(x_filtered[i]), endpoint=False)
    plt.plot(t_new, x_filtered[i])

plt.suptitle('Exercitiul 4')
plt.legend(["original", "w=4", "w=10", "w=16", "w=25"], loc ="lower right")
plt.tight_layout()
plt.show()

# c)

f_sample = 1/3600

f_cutoff = f_sample/6
f_nyquist = 0.5 * f_sample
f_normalized = f_cutoff / f_nyquist

print("Frecventa cutoff in hz:", f_cutoff)
print("Frecventa cutoff normalizata:", f_normalized)

# d)

f_order = 5
rp = 5
print("f_normalized: ", f_normalized)

butter_b, butter_a = butter(f_order, f_normalized, btype="low")
cheby_b, cheby_a = cheby1(f_order, rp, f_normalized, btype="low")

w_butter, h_butter = freqz(butter_b, butter_a)
w_cheby, h_cheby = freqz(cheby_b, cheby_a)

plt.figure(figsize=(10, 6))
plt.plot(w_butter, 20 * np.log10(np.abs(h_butter)))
plt.plot(w_cheby, 20 * np.log10(np.abs(h_cheby)))

plt.xlabel("Frecventa (Hz)")
plt.ylabel("Amplitudine")
plt.title("Raspuns al filtrelor")
plt.legend(["Butter", "Cheby"], loc ="lower left")
plt.grid(True)
plt.tight_layout()
plt.show()

# e)

x_filtrat_butter = filtfilt(butter_b, butter_a, x_sliced)
x_filtrat_cheby = filtfilt(cheby_b, cheby_a, x_sliced)

plt.plot(x_sliced)
plt.plot(x_filtrat_butter)
plt.plot(x_filtrat_cheby)
plt.xlabel("Frecventa (Hz)")
plt.ylabel("Amplitudine")
plt.legend(["Original", "Butter", "Cheby"], loc ="lower right")
plt.title("Semnalul dupa trecerea prin filtre")
plt.grid(True)
plt.tight_layout()
plt.show()

# f)

f_orders = [3, 5, 7]
rps = [1, 10, 15]

for i, order in enumerate(f_orders):
    butter_b, butter_a = butter(order, f_normalized, btype="low")
    x_filtrat_butter = filtfilt(butter_b, butter_a, x_sliced)

    plt.subplot(3, 1, i+1)
    plt.plot(x_sliced, label="original")
    plt.plot(x_filtrat_butter, label=f"butter, ordin={order}")

    for new_rp in rps:
        cheby_b, cheby_a = cheby1(order, new_rp, f_normalized, btype="low")
        x_filtrat_cheby = filtfilt(cheby_b, cheby_a, x_sliced)

        plt.plot(x_filtrat_cheby, label=f"Cheby, rp={new_rp}")

    plt.suptitle("Semnale filtrate, ordin = 3, 5, 7")
    plt.tight_layout()
    plt.xlabel("Frecventa (Hz)")
    plt.ylabel("Amplitudine")
    plt.grid(True)
    plt.legend()

plt.show()

