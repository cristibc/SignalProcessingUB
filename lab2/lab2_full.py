import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time
from scipy.io import wavfile

freq = 2
amp = 1.5
faza = 6
samples = 1000

t = np.arange(0, 1, 1/samples)
sinusoidala = amp * np.sin(2 * np.pi * freq * t + faza)

cosinus = amp * np.cos(2 * np.pi * freq * t + faza - np.pi/2)

plt.figure(figsize = (8, 8))
plt.subplot(211)
plt.plot(t, sinusoidala)
plt.grid(True)
plt.title("Sinusoidala")
plt.subplot(212)
plt.plot(t, cosinus)
plt.grid(True)
plt.title("Cosinus")

plt.show()

freq2 = 2
amp2 = 1
faza2 = 6
samples2 = 1000

t2 = np.arange(0, 1, 1/samples2)
def sinusoidala2(faza): 
    return amp * np.sin(2 * np.pi * freq2 * t2 + faza)

plt.plot(t2, sinusoidala2(0))
plt.plot(t2, sinusoidala2(45))
plt.plot(t2, sinusoidala2(90))
plt.plot(t2, sinusoidala2(135))
plt.grid(True)
plt.title("Ex 2 - Partea 1")
plt.show()

SNR = [0.1, 1, 10, 100]
faze = [0, 45, 90, 135]

def gamaSNR(faza, sample):
    x = sinusoidala2(faza)
    gaus = np.random.normal(size=sample)
    gamasq = np.linalg.norm(x)**2 / np.linalg.norm(gaus)**2 / 10
    gama = np.sqrt(gamasq)
    sinNoised = amp * np.sin(2 * np.pi * freq2 * t2 + faza) + gama * gaus
    return sinNoised

# Test for 1 wave
# plt.plot(t2, gamaSNR(0, 1000))
# plt.show()

all_signals = []

fig, axs = plt.subplots(4)
for i in range(len(SNR)):
    combined_signal = np.zeros(samples2)
    for faza in faze:
        x = sinusoidala2(faza)
        gaus = np.random.normal(size=samples2)
        gamasq = np.linalg.norm(x)**2 / np.linalg.norm(gaus)**2 / SNR[i]
        gama = np.sqrt(gamasq)
        sinNoised = amp * np.sin(2 * np.pi * freq2 * t2 + faza) + gama * gaus
        axs[i].plot(t2, sinNoised)
        combined_signal += sinNoised
    all_signals.append(combined_signal)

fig.suptitle('Ex 2 - Partea 2')
plt.show()

# Ex 3
sample1 = 1600
t1 = np.linspace(0, 1, sample1, endpoint=False)
def sine1(hz, time):
    return np.sin(2 * np.pi * hz * time)
sample2 = 16000 * 3
t2 = np.linspace(0, 3, sample2, endpoint=False)
def sine2(hz, time):
    return np.sin(2 * np.pi * hz * time)
t3 = np.linspace(0, 1, 48000, endpoint=False)
def sawtooth(hz, time):
    return 2 * (time * hz - np.floor(time * hz + 0.5))
t4 = np.linspace(0, 1, 60000, endpoint=False)
def square(hz, time):
    return np.sign(np.sin(2* np.pi * hz * time))

sd.play(sine1(400, t1) * 0.01, 1600)
time.sleep(1)
sd.play(sine2(800, t2) * 0.01, 1600)
time.sleep(1)
sd.play(sawtooth(240, t3) * 0.002, 48000)
time.sleep(1)
sd.play(square(300, t4) * 0.002, 60000)
time.sleep(1)
sd.stop()

wavfile.write("sawtooth.wav", 48000, (sawtooth(240, t3) * 0.002).astype(np.float32))
samplerate, data = wavfile.read("sawtooth.wav")
sd.play(data, samplerate)
time.sleep(1)
sd.stop()

# Ex 4

fig, ax = plt.subplots(3)
fig.suptitle('Exercitiul 4')

ax[0].plot(t2, sine2(800, t2))
ax[0].set_xlim([0, 0.02])
ax[0].axhline(y=0, color='k')

ax[1].plot(t3, sawtooth(240, t3))
ax[1].set_xlim([0, 0.02])
ax[1].axhline(y=0, color='k')

combinat = sine2(800, t2) + sawtooth(240, t3)
ax[2].plot(t2+t3, combinat)
ax[2].set_xlim([0, 0.02])
ax[2].axhline(y=0, color='k')

plt.show()

# Ex 5
freq1 = 800
freq2 = 16000

t = np.linspace(0, 1, sample2, endpoint=False)
signal1 = sine2(freq1, t)
signal2 = sine2(freq2, t)

semnal_final = np.concatenate((signal1, signal2))
sd.play(semnal_final * 0.002, sample2)
sd.wait()

# Observam ca semnalul cu frecventa mai mica suna mai low, cel cu frecventa mai mare suna mai inalt

# Ex 6

Fs = 44100

freq_a = Fs / 2
freq_b = Fs / 4
freq_c = 0

semnal_a = sine2(freq_a, t)
semnal_b = sine2(freq_b, t)
semnal_c = sine2(freq_c, t)

fig, ax = plt.subplots(3)
fig.suptitle('Exercitiul 6')

ax[0].plot(t, semnal_a)
ax[0].set_xlim([0, 0.002])
ax[0].axhline(y=0, color='k')

ax[1].plot(t, semnal_b)
ax[1].set_xlim([0, 0.002])
ax[1].axhline(y=0, color='k')

ax[2].plot(t, semnal_c)
ax[2].set_xlim([0, 0.002])
ax[2].axhline(y=0, color='k')

plt.show()

# Semnalul a) are o perioada completa, semnalul b) are 4 perioade si semnalul c) are un semnal constant dar o amplitudine foarte mica

# Ex 7)
sample = 1000
t = np.linspace(0, 2, sample, endpoint=False)
semnal_original = np.sin(2 * np.pi * 100 * t)

decimat1 = semnal_original[::4]
decimat2 = semnal_original[1::4]

fig, ax = plt.subplots(3)
fig.suptitle('Exercitiul 7, a) si b)')

ax[0].plot(t, semnal_original)
ax[0].set_xlim([0, 0.5])
ax[0].axhline(y=0, color='k')

ax[1].plot(t[::4], decimat1)
ax[1].set_xlim([0, 0.5])
ax[1].axhline(y=0, color='k')

ax[2].plot(t[1::4], decimat2)
ax[2].set_xlim([0, 0.5])
ax[2].axhline(y=0, color='k')
plt.show()

# Diferenta intre semnalul original si cele decimate este ca cele decimate se repeta de mai putine ori
# Si diferenta intre cele decimate este ca sunt defazate unul fata de celalalt

# Ex 8 

alfa = np.linspace(-np.pi/2, np.pi/2, 100)
sin_alfa = np.sin(alfa)
eroare = np.abs(sin_alfa - alfa)

fig, ax = plt.subplots(2,2)
fig.suptitle('Exercitiul 8')
ax[0, 0].plot(alfa, alfa)
ax[0, 0].plot(alfa, sin_alfa)
ax[0, 0].title.set_text('Alfa / Sin(Alfa)')
ax[0, 0].legend(["alfa", "sin(alfa)"], loc ="lower right") 
# ax[0].errorbar(alfa, sin_alfa, yerr=np.abs(sin_alfa-alfa))
ax[0, 1].plot(alfa, eroare)
ax[0, 1].title.set_text('Eroare Alfa/Sin(Alfa)')
plt.grid(True)

pade = (alfa - (7 * alfa**3) / 60) / (1 + (alfa**2/20))

ax[1, 0].plot(alfa, alfa)
ax[1, 0].plot(alfa, pade)
ax[1, 0].title.set_text('Alfa / Pade')
ax[1, 0].legend(["alfa", "pade"], loc ="lower right") 

ax[1, 1].plot(alfa, alfa)
ax[1, 1].plot(alfa, pade)
ax[1, 1].title.set_text('Alfa / Pade Logaritmic')
ax[1, 1].legend(["alfa", "pade"], loc ="lower right") 

fig.tight_layout()
plt.yscale("log")
plt.xlim(0, 1.6)
plt.show()
