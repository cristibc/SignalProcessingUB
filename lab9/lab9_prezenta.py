import math
import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt


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


model = SimpleExpSmoothing(time_series)
fit1 = model.fit(smoothing_level=.2)
fvalues = fit1.fittedvalues
pred1 = fit1.forecast(1)

plt.plot(t, time_series)
plt.plot(t, fvalues)

plt.tight_layout()
plt.show()

def smoothing(t, x, alpha):
    N = len(x)
    i = np.arange(N)
    print(i)
    S = alpha * ((1-alpha)**(t - i)) * x + ((1 - alpha) ** t ) * x[0]
    print(S)
    return S

x_new = smoothing(t, time_series, 0.2)
plt.plot(t, fvalues)
plt.plot(t, x_new)
plt.show()

