import pandas as pd
from typing import List
from .filters import ButterworthFilter, ChebyshevFilter, Envelope_MA


def preprocess_data(data: pd.DataFrame, features_to_smoothe: List[str], do_smoothe=False, keep_nans=False):
    """
        This is the main preprocessing function.
        @param data: Data
        @param features_to_smoothe: Names of features for smoothing
        @param do_smoothe: Activation for smoothing
        @param keep_nans: keep NaN values during smoothing operation
        @return: pre-processed data
    """
    if do_smoothe:
        data = smoothe_data(data, features_to_smoothe, keep_nans)

    data = data.astype('float')
    data = data.dropna(axis=0)
    return data


def demod_signals(data: pd.DataFrame, labels: List[str], keep_nans=True, T_mod=20):
    """
       Demodulate signals
        @param data: Data
        @param labels: Labels of features to demodulate
        @param keep_nans: Keep NaN values during demodulation
        @param T_mod: Period of signal to remove in samples
        @return: dataframe containing demodulated values
    """
    filt_labels = [f'{label}_filt' for label in labels]
    env_labels = [f'{label}_env' for label in labels]
    data[filt_labels] = ChebyshevFilter(T=T_mod * 10, keep_nans=keep_nans, remove_offset=True).fit_transform(data[labels])
    data[env_labels] = Envelope_MA(T=T_mod, keep_nans=keep_nans, remove_offset=True).fit_transform(data[labels])
    return data


def smoothe_data(data: pd.DataFrame, features_to_smoothe: List[str], keep_nans=False):
    """
    Smoothe signals through Butterworth filter
    @param data: Data
    @param features_to_smoothe: Labels of features to smoothe
    @param keep_nans: Keep NaN values during smoothing
    @return: dataframe containing smoothed values
    """
    filter = ButterworthFilter(order=2, T=10, keep_nans=keep_nans)
    filtered_vals = filter.fit_transform(data[features_to_smoothe].to_numpy())
    data[features_to_smoothe] = filtered_vals
    return data


