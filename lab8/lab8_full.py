import math
import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy
import time
import csv
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# a)

N = 1000
t = np.arange(N)
trend = 0.5 * (t/30)**2
seasonality = 10 * np.sin(2 * np.pi * t/50) + 5 * np.sin(2 * np.pi * t/20)
noise = np.random.normal(0, 10, N)

time_series = trend + seasonality + noise

plt.figure(figsize=(12, 6))
plt.subplot(4, 1, 1)
plt.plot(t, time_series, label="Seria de timp")
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t, trend, label="Trend")
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t, seasonality, label="Sezon")
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t, noise, label="Noise")
plt.legend()

plt.tight_layout()
plt.show()

# b)

autocorr = np.correlate(time_series, time_series, mode="full")
autocorr = autocorr[N-1:] / np.max(autocorr)
plt.stem(autocorr[:50])
plt.title("Autocorelatie")
plt.show()

# c)

p = 100
train_size = int(0.6 * N)
train, test = time_series[:train_size], time_series[train_size:]

model = AutoReg(train, lags=p)
model_fit = model.fit()

prediction = model_fit.predict(start=len(train), end=len(train)+len(test)-1)

plt.plot(t, time_series, label="Actual")
plt.plot(t[len(train):], prediction, label="Predicted")
plt.legend()
plt.show()