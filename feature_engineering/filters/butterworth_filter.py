from scipy import signal as sig

from .filter import Filter


class ButterworthFilter(Filter):
    """
    Butterworth filter for data smoothing.
    """
    T = 10
    order = 2
    filter_type = 'lowpass'

    def __init__(self, remove_offset=False, keep_nans=False, T=10, order=2, filter_type='lowpass', **kwargs):
        super().__init__(remove_offset=remove_offset, keep_nans=keep_nans)
        self._set_attrs(T=T, order=order, filter_type=filter_type)

    def calc_coef(self, X, y=None, **fit_params):
        return sig.butter(self.order, 1 / self.T, btype=self.filter_type)