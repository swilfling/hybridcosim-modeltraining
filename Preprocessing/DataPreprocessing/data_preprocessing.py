from . Filter import ButterworthFilter, ChebyshevFilter, Envelope_MA


def preprocess_data(data, features_to_smoothe, do_smoothe=False, keep_nans=False):
    if do_smoothe:
        data = smoothe_data(data, keep_nans, features_to_smoothe)
    return data


def demod_signals(df, labels, keep_nans=True, T_mod=20):
    '''
        Process modulated signals
    '''
    filt_labels = [f'{label}_filt' for label in labels]
    env_labels = [f'{label}_env' for label in labels]
    df[filt_labels] = ChebyshevFilter(T=T_mod * 10, keep_nans=keep_nans, remove_offset=True).fit_transform(df[labels])
    df[env_labels] = Envelope_MA(T=T_mod, keep_nans=keep_nans, remove_offset=True).fit_transform(df[labels])
    return df


def smoothe_data(data, keep_nans, features_to_smoothe):
    # Data Smoothing
    if smoothe_data:
        filter = ButterworthFilter(order=2, T=10, keep_nans=keep_nans)
        filtered_vals = filter.fit_transform(data[features_to_smoothe].to_numpy())
        data[features_to_smoothe] = filtered_vals
    return data


