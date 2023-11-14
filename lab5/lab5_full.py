import math
import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time
import csv

# Ex 1

# a)

x = np.genfromtxt('lab5/Train.csv', delimiter=',', skip_header=1, usecols=2)
print(x)
# x = x - np.mean(x)
X = np.fft.fft(x)

print(X)

N = len(x)
N_halved = int(N/2)
X = abs(X/N)

X = X[:N_halved]

Fs = 1 / 3600
f = Fs * np.linspace(0, N_halved, N_halved)/N

# b)
print("Intervalul de timp este de: ", int(N/24), " zile")

# c) 

fmax = 0.5 * Fs
print("Frecventa maxima: ", fmax)
t = np.arange(N_halved)

# d)

plt.plot(f, X)
plt.suptitle('Exercitiul d)')
plt.show()

# e)
print("Semnalul prezinta o componenta continua, cea care este egala cu 0Hz")
X[0] = 0

# f)
poz = np.argsort(X)[-4:]

freqs = f[poz]

print("Cele mai mari 4 valori:")
for i, index in enumerate(poz):
    print(f"Valoare {i + 1}: {X[index]}, Frecventa: {freqs[i]} Hz")

# g)

x_cut = x[3408:3408+730]
f_cut = np.arange(len(x_cut))

plt.plot(f_cut, x_cut)
plt.suptitle('Exercitiul g)')
plt.show()

# h) 
# Am putea localiza data la care a inceput aceasta masurare comparand pattern-urile din semnal astfel:
# Determinam aproximativ ziua analizand perioade care coincid cu evenimente din viata reala
# Spre ex: craciun, unde avem trafic major pana in ziua sarbatorii, urmat de un drop semnificativ
# Si apoi comparam aceste date fixe cu zilele saptamanii (stim ca in weekend este mai mult trafic)
# Determinand astfel anul in care suntem (calendarul nostru se repeta o data la 28 de ani)
# Si mai apoi data in care incepe masurarea prin calcule simple
# Deci atata timp cat avem un dataset care contine date pentru mai mult de un an, putem determina si inceputul masurarilor

# i) Afisam doar valorile care nu deviaza cu mai mult de 100% peste media tuturor valorilor (sarbatori?)


high_limit = np.mean(X) * 2
mask = X <= high_limit
print("meanX: ", np.mean(X))

plt.plot(f[mask], X[mask])
plt.suptitle('Exercitiul i)')
plt.show()
