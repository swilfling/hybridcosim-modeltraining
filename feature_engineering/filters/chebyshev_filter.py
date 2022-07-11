from scipy import signal as sig

from .filter import Filter


class ChebyshevFilter(Filter):
    """
       Chebyshev filter for data smoothing.
    """
    T = 10
    order = 2
    ripple = 0.1
    filter_type = 'lowpass'

    def __init__(self, T=10, order=2, ripple=0.1, filter_type='lowpass', remove_offset=False, keep_nans=True,
                 **kwargs):
        super().__init__(remove_offset=remove_offset, keep_nans=keep_nans)
        self._set_attrs(T=T, order=order, ripple=ripple, filter_type=filter_type)

    def calc_coef(self, X, y=None, **fit_params):
        return sig.cheby1(self.order, self.ripple, 1 / self.T, btype=self.filter_type)