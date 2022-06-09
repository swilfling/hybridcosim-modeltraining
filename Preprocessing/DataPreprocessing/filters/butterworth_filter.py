from scipy import signal as sig

from .filter import Filter


class ButterworthFilter(Filter):
    """
    Butterworth lowpass filter for data smoothing.
    """
    T = 10
    order = 2

    def __init__(self, T=10, order=2, **kwargs):
        super().__init__(**kwargs)
        self._set_attrs(T=T, order=order)

    def _fit(self, X, y=None, **fit_params):
        return sig.butter(self.order, 1 / self.T)