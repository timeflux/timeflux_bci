import numpy as np
from scipy.signal import filtfilt
from sklearn.base import BaseEstimator, TransformerMixin
from timeflux_dsp.utils.filters import construct_iir_filter

from sklearn.base import BaseEstimator, TransformerMixin
from timeflux_dsp.utils.filters import construct_iir_filter


class FiltFilt(BaseEstimator, TransformerMixin):
    """Estimation of mega-raw
    Parameters
    ----------
    frequencies : tuple
        The center frequencies of the  bandpass filters
    band_width : float (default 0.2)
        The width of the bandpass frequencies
    rate : float | None, (default None)
        The sampling frequency of the signal.
    truncate: bool
    filtfilt_kwargs: dict | None
    design_kwargs: dict | None
    """

    def __init__(self, frequencies, band_width, rate=None, truncate=True, filtfilt_kwargs=None, design_kwargs=None):
        """Init."""
        self._truncate = truncate
        self._design_kwargs = design_kwargs or {}
        self._design_kwargs['output'] = 'ba'
        self._filtfilt_kwargs = filtfilt_kwargs or {}
        self._length = None
        self._filterbank = []
        for frequency in frequencies:
            ba, _ = construct_iir_filter(rate or 1.0, [frequency - band_width / 2, frequency + band_width / 2],
                                         'bandpass',
                                         **self._design_kwargs)
            self._filterbank.append(ba)
            if self._length is None:
                self._length = len(ba[0])

    def fit(self, X, y=None):
        """Fit.
        Do nothing. For compatibility purpose.
        Parameters
        ----------
        X : ndarray, shape (n_trials,n_channels, n_samples )
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.
        Returns
        -------
        self : FilterBank instance
            The FilterBank instance.
        """
        return self

    def transform(self, X):
        """Estimate the mega-signal
        Parameters
        ----------
        X : ndarray, shape (n_trials,  n_channels * n_frequencies,  n_samples - filter_length)
            ndarray of trials.
        Returns
        -------
        mega_X : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
            ndarray of covariance matrices for each trials and for each
            frequency bin.
        """
        out = []
        for _ba in self._filterbank:
            X_filtered = filtfilt(*_ba, X, **self._filtfilt_kwargs)
            out.append(X_filtered)
        mega_X = np.concatenate(out, axis=1)
        if self._truncate:
            mega_X = mega_X[:, :, self._length:-self._length]
        return mega_X