import numpy as np
import matplotlib.pyplot as plt

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
plt.title("Sinusoidala")
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

plt.plot(t2, gamaSNR(0, 1000))
plt.show()

for i in SNR:
    for faza in faze:
        x = sinusoidala2(faza)
        gaus = np.random.normal(size=1000)
        gamasq = np.linalg.norm(x)**2 / np.linalg.norm(gaus)**2 / SNR[i]
        gama = np.sqrt(gamasq)
        sinNoised = amp * np.sin(2 * np.pi * freq2 * t2 + faza) + gama * gaus
    

fig, axs = plt.subplots(4)
axs[0].plot(t2, gamaSNR(0, 1000))
axs[0].plot(t2, gamaSNR(45, 1000))
axs[0].plot(t2, gamaSNR(90, 1000))
axs[0].plot(t2, gamaSNR(135, 1000))
plt.show()