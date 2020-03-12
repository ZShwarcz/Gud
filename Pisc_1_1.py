import numpy as np
from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

FD =  5  #частота дискретизации, отсчётов в секунду
N = 300    # длина входного массива

sig = np.array([(np.sin(2*np.pi*10.0*t) + np.sin(2*np.pi*20.0*t) + 2) *
                 np.sin(2*np.pi*100.0*t) + 2 *
                 np.sin(2*np.pi * 80.0 * t) for t in range(1, N)])


spectrum = rfft(sig)

plt.plot(sig)
plt.show()
plt.plot(spectrum)
plt.show()
