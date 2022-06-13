from matplotlib import pyplot as plt


def plot_spectra(data, fft_data):
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_xlabel('Time in Number of Samples')
    ax1.set_title("Time Series - Zero Mean")
    ax1.plot(data)
    ax2.set_title("Spectrum - Magnitude Response")
    ax2.stem(fft_data[0], fft_data[1])
    ax2.set_xlabel("Frequency normalized over sampling frequency f_s")
    plt.tight_layout()
    plt.show()


def plot_ac(xcorr, psd):
    fig2, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_title("Auto-correlation - Normalized")
    ax1.plot(xcorr)
    ax2.set_title("Power Spectral Density")
    plt.plot(psd)
    plt.tight_layout()
    plt.show()