from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import numpy as np

N = 1000
FD = 10000

# Обычный синус

pure_sig = np.array([np.sin(2.*np.pi*50.0*t/FD) for t in range(N)]) #сигнал

#plt.plot(pure_sig)
#plt.title("График синуса")
#plt.xlabel("Время, [t]")
#plt.ylabel("Амплитуда")
#plt.show()

# Спектр синуса
spectrum = rfft(pure_sig) #сигнал спектр

#plt.plot(spectrum)
#plt.title("Спектр синуса")
#plt.xlabel("частота, [Hz]")
#plt.ylabel("Амплитуда")
#plt.show()

# Белый шум
mean = 0
std = 1
num_samples = 1000
samples = np.random.normal(mean, std, size=num_samples)

white_noise = pure_sig + samples

plt.plot(white_noise)
plt.title("График сигнала с белым шумом")
plt.xlabel("Время, [t]")
plt.ylabel("Амплитуда")
plt.show()

white_noise_spec = rfft(white_noise)

plt.plot(np.arange(N), white_noise_spec)
plt.title("Спектр сигнала с белым шумом")
plt.xlabel("частота, [Hz]")
plt.ylabel("Амплитуда")
plt.show()

sma = np.convolve(white_noise, np.ones(100), 'valid')

plt.plot(sma)
plt.title("График кользящей средней с окном = 100")
plt.xlabel("Время, [t]")
plt.ylabel("Амплитуда")
plt.show()

sma_spec = rfft(sma)

plt.plot(np.arange(N), sma_spec)
plt.title("Спектр кользящей средней с окном = 100")
plt.xlabel("частота, [Hz]")
plt.ylabel("Амплитуда")
plt.show()


