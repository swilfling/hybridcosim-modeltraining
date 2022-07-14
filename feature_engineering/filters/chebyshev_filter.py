from scipy import signal as sig

from . import FilterDirectForm


class ChebyshevFilter(FilterDirectForm):
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

    def _get_feature_names_out(self, feature_names=None):
        return [f'{feat}_cheb_{self.T}' for feat in feature_names]