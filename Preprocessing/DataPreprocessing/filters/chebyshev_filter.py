from scipy import signal as sig

from .filter import Filter


class ChebyshevFilter(Filter):
    """
       Chebyshev lowpass filter for data smoothing.
    """
    T = 10
    order = 2
    ripple = 0.1

    def __init__(self, T=10, order=2, ripple=0.1, **kwargs):
        super().__init__(**kwargs)
        self._set_attrs(T=T, order=order, ripple=ripple)

    def _fit(self, X, y=None, **fit_params):
        return sig.cheby1(self.order, self.ripple, 1 / self.T)