from scipy import signal as sig

from .filter import Filter


class ButterworthFilter(Filter):
    """
    Butterworth filter for data smoothing.
    """
    T = 10
    order = 2
    filter_type = 'lowpass'

    def __init__(self, T=10, order=2, filter_type='lowpass', **kwargs):
        super().__init__(**kwargs)
        self._set_attrs(T=T, order=order, filter_type=filter_type)

    def _fit_model(self, X, y=None, **fit_params):
        return sig.butter(self.order, 1 / self.T, btype=self.filter_type)