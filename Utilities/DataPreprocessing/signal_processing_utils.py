import pandas as pd
import numpy as np
import scipy.signal as sig
from sklearn import preprocessing


# Threshold data - remove values in curves where switch is off
def threshold_data(df, curve, switch):
    curve_data = (df[curve].copy()).where(df[switch] > 0)
    data_pruned = pd.concat([curve_data, df[switch]])
    return curve_data, data_pruned

def thresh_max(df, thresh_val, new_val=0.0):
    df_new = df.copy()
    df_new[df_new > thresh_val] = new_val
    return df_new


def thresh_min(df, thresh_val, new_val=0.0):
    df_new = df.copy()
    df_new[df_new < thresh_val] = new_val
    return df_new


def thresh_abs(df, thresh_val, new_val=0.0):
    df_new = df.copy()
    df_new[np.abs(df_new) > thresh_val] = new_val
    return df_new


def remove_offset(signal):
    offset = np.mean(signal)
    signal_zeromean = signal - offset
    return signal_zeromean, offset


def diff(df, periods=1):
    data = df.diff(periods)
    return data.fillna(0)


def filter_signal(signal, T_to_filter, keep_nans=True):
    mask = signal.isna()
    signal = np.nan_to_num(signal)
    signal_zeromean, offset = remove_offset(signal)
    f_cutoff = 1 / T_to_filter / 10
    b_cheb, a_cheb = sig.cheby1(2, 0.1, f_cutoff)
    y = sig.lfilter(b_cheb, a_cheb, signal_zeromean) + offset
    # Plot filter frequency response
    # w, freqres = sig.freqs(b_cheb,a_cheb)
    # plt.figure()
    # plt.semilogx(np.abs(freqres))
    # plt.title('Chebyshev filter - frequency response')
    # plt.show()
    if keep_nans:
        y[mask==True] = np.nan
    return y


# TODO fix this....
def lowpass_fft(data, fcut=0.5):
    offset = np.mean(data)
    df_normalized = data - offset
    num_samples = len(data)
    fft = np.fft.fft(df_normalized)
    f_cutoff_samples_passband = int(num_samples / 2 * fcut)
    f_cutoff_samples_stopband = int(num_samples / 2 * (1-fcut))
    lowpass_passband = np.ones((f_cutoff_samples_passband,1))
    lowpass_stopband = np.zeros((f_cutoff_samples_stopband,1))
    lowpass_half = np.vstack((lowpass_passband, lowpass_stopband))
    lowpass = np.vstack((lowpass_half, np.flipud(lowpass_half)))
    fft = np.reshape(fft,(num_samples,1))
    fft_thresholded = fft * lowpass

    df_fftthresh = np.fft.ifft(fft_thresholded)
    return df_fftthresh + offset



def fft_abs(df_normalized, spectrum_range=1):
    num_samples = len(df_normalized)
    num_samples_to_plot = int(num_samples * spectrum_range / 2)
    fft_abs = np.abs(np.fft.fft(df_normalized))
    fft_abs = fft_abs[:num_samples_to_plot]
    fft_abs_normalized = fft_abs / num_samples
    #fft_frequencies = np.fft.fftfreq(n=num_samples_to_plot * 2)
    fft_freqs = np.linspace(0, spectrum_range / 2, num_samples_to_plot)
    # fft_np = np.fft.fftshift(fft_np)
    return fft_freqs, fft_abs_normalized


def fft_threshold(data, threshold_factor=0.1):
    offset = np.mean(data)
    df_normalized = data - offset

    fft = np.fft.fft(df_normalized)
    abs_fft = np.abs(fft)
    fft_thresholded = fft * (abs_fft > threshold_factor * max(abs_fft))

    df_fftthresh = np.fft.ifft(fft_thresholded)
    return df_fftthresh + offset


# This is taken from https://stackoverflow.com/a/69357933
def calculate_envelope(signal, T, maxmin='max'):
    # analytic_sig = sig.hilbert(signal)
    # envelope = np.abs(analytic_sig)
    signal = pd.DataFrame(signal)
    if maxmin == 'max':
        env = signal.rolling(window=T).max().shift(int(-T / 2))
    else:
        env = signal.rolling(window=T).min().shift(int(-T/2))
    return np.nan_to_num(np.array(env))


def xcorr_psd(data):
    xcorr = np.correlate(data, data, 'same') / len(data)
    _, psd = sig.periodogram(data)
    return xcorr, psd / len(data)


def filter_smoothe_signal(signal, T=100):
    b, a = sig.butter(2, 2 * np.pi * 1 / T)
    return sig.lfilter(b, a, signal)


def remove_nans(signal):
    mask = signal.isna()
    signal = np.nan_to_num(signal)
    return signal, mask


def apply_nans(signal, mask):
    signal[mask is True] = np.nan


def calc_avg_signals(signal_1, signal_2):
    N_samples = len(signal_1)
    assert(len(signal_1) == len(signal_2))
    signals_stacked = np.hstack((np.reshape(signal_1, (N_samples, 1)), np.reshape(signal_2, (N_samples, 1))))
    return np.average(signals_stacked, axis=1)


def calc_avg_envelope(signal, T, do_filter=True, keep_nans=True):
    signal_zeromean, offset = remove_offset(signal)
    signal_zeromean, mask_missing_values = remove_nans(signal_zeromean)

    # Calculate high and low envelope
    envelope_h = calculate_envelope(signal_zeromean, T, 'max') + offset
    envelope_l = calculate_envelope(signal_zeromean, T, 'min') + offset
    # Average both
    envelope_avg = calc_avg_signals(envelope_h, envelope_l)
    if do_filter:
        envelope_avg = filter_smoothe_signal(signal)
    if keep_nans:
        apply_nans(envelope_avg, mask_missing_values)
    return envelope_avg, envelope_h, envelope_l


def prune_data(df, curves, switches):
    num_curves = len(curves)
    num_switches = len(switches)
    # If only one switch - repeat switches
    switches_new = [switches[0] for i in range(num_curves)] if num_switches == 1 else switches
    #print(switches_new)
    list_data = [threshold_data(df, curve, switch)[0] for curve, switch in zip(curves,switches_new)]
    data_pruned = pd.concat(list_data, axis=1, names=curves)
    data_pruned_full = pd.concat([data_pruned,df[switches]])
    return data_pruned_full


def scale_curve(df, label):
    min_max_scaler = preprocessing.MinMaxScaler()
    df[label + '_scaled'] = (min_max_scaler.fit_transform(pd.DataFrame(df[label])))



def calc_hists(df):
    hists = []
    for column in df.columns:
        hist, bins = calc_hist(df[column])
        hists.append(pd.DataFrame(hist, columns=[column], index=bins[:-1]))
    return hists

def calc_hist(data):
    return np.histogram(data,range=(np.min(data),np.max(data)))