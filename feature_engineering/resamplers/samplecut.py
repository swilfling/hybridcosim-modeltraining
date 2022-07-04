from imblearn.base import BaseSampler


class SampleCut(BaseSampler):
    num_samples = 0
    _sampling_type = 'bypass'

    def __init__(self, num_samples=0, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples

    def _fit_resample(self, X, y=None):
        if y is not None:
            return X[self.num_samples:], y[self.num_samples:]
        else:
            return X[self.num_samples:], None

    def transform(self, X):
        return X[self.num_samples:]

