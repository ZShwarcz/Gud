import numpy as np
from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

FD = 10000 # частота дискретизации, отсчётов в секунду
N = 1000 # длина входного массива, 0.091 секунд при такой частоте дискретизации
pure_sig = np.array([np.sin(2.*np.pi*50.0*t/FD) for t in range(N)])
print(len(pure_sig))
spectrum = rfft(pure_sig)
print(len(spectrum))
mean = 0
std = 1
num_samples = 1000
samples = np.random.normal(mean, std, size=num_samples)

plt.plot(samples)
plt.show()

white_noise = pure_sig + samples

print(len(white_noise))

plt.plot(np.arange(N), white_noise, 'r')
plt.xlabel(u'Время, c')
plt.ylabel(u'Высота')
plt.title(u'Сигнал 50 Гц')
plt.grid(True)
plt.show()

