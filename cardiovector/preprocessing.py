import pywt
import wfdb
import numpy as np

from ._lib import copy_record


def choose_wavelet(signal_length, fallback=None):
    """
    Make a decision on to which wavelet should be used for BW filtering. If no decision can be made, due to not enough
    data samples, `fallback` will be returned if available.

    Currently there are two wavelets considered, both at 10th level of decomposition: symmlet-10 and daubechies-2.
    Both were proven effective for BW filtering (@Bunluechokchai2010, @Maheshwari16).

    Parameters
    ----------
    signal_length : int
        Number of samples in the signal to process.
    fallback : (wavelet : str, level : int), optional
        Tuple containing fallback wavelet name and desired decomposition level to return of none of wavelets included
        in the package are available.

    Returns
    -------
    wavelet_level : (wavelet : str, level : int)
        Tuple containing name and desired level of selected wavelet to use for filtering.
    """

    for (wavelet, required_level) in _wavelets:
        max_level = pywt.dwt_max_level(signal_length, pywt.Wavelet(wavelet))
        if max_level >= required_level:
            return wavelet, required_level

    if fallback is None:
        raise ValueError('Could not find a wavelet that would be good enough. Sample is too small!')

    return fallback


def remove_baseline_wandering_s(signal, wavelet_level=None):
    """
    Filter BW from a array-like signal using a DWT algorithm. Wavelet to use is chosen using `choose_wavelet` function,
    unless custom is provided in `wavelet_level` arg.

    Parameters
    ----------
    signal : (N,) array_like
        ECG signal to filter.
    wavelet_level : tuple (wavelet : str, level : int), optional
        Tuple containing fallback wavelet name and desired decomposition level to use for filtering.

    Returns
    -------
    out_signal : (N,) numpy.array
        Filtered signal.
    """

    (wavelet, level) = _wavelet_for_signal(len(signal), wavelet_level)

    ca10_orig, *cDn = pywt.wavedecn(signal, wavelet)
    ca10 = np.zeros(ca10_orig.shape)

    result = pywt.waverecn([ca10, *cDn], wavelet)
    so = result.shape[0]
    si = signal.shape[0]
    if so > si:
        n = si - so
        result = result[:n]

    return result


def remove_baseline_wandering(record, wavelet_level=None):
    """
    Filter BW from a `wfdb.Record` instance using a DWT algorithm. Wavelet to use is chosen using
    `choose_wavelet` function, unless custom is provided in `wavelet_level` arg.

    Parameters
    ----------
    record : wfdb.Record
        WFDB Record to filter.
    wavelet_level : tuple (wavelet : str, level : int), optional
        Tuple containing fallback wavelet name and desired decomposition level to use for filtering.

    Returns
    -------
    out_record : wfdb.Record
        Filtered record.
    """

    if not isinstance(record, wfdb.Record):
        raise ValueError('Not a wfdb.Record. For sigle signals, use remove_baseline_wandering_s.')

    if record.d_signal is None:
        raise ValueError('Record should have digital signals')

    record = copy_record(record)

    data_len = record.sig_len
    (wavelet, level) = _wavelet_for_signal(data_len, wavelet_level)

    for i in range(record.n_sig):
        signal = record.d_signal[:, i]
        record.d_signal[:, i] = remove_baseline_wandering_s(signal, (wavelet, level))

    record.comments.append('Filtering: removed base wandering')
    return record


def _wavelet_for_signal(data_len, fallback):
    if fallback is None:
        return choose_wavelet(data_len)
    else:
        return fallback


_wavelets = [('sym10', 10), ('db2', 10)]


def _slice_if_present(sig, sampfrom, sampto):
    if sig is not None and sig.shape != ():
        return sig[sampfrom:sampto, :]
    else:
        return sig


def recslice(record, sampfrom=None, sampto=None):
    """
    Slice the record to leave only a part of the signal.

    Parameters
    ----------
    record : wfdb.Record
        Record to modify.
    sampfrom : int, optional
        First sample (inclusive) to include in modified record.
    sampto : int, optional
        Last sample (exclusive) of the modified record.
    Returns
    -------
    record : wfdb.Record
        The modified record.
    """

    if not isinstance(record, wfdb.Record):
        raise ValueError('Not a wfdb.Record.')

    record = copy_record(record)

    if sampfrom is None and sampto is None:
        raise ValueError('sampfrom or sampto not defined. Call would have no effect')

    if sampfrom is None:
        sampfrom = 0

    if sampto is None:
        sampto = record.siglen

    record.p_signal = _slice_if_present(record.p_signal, sampfrom, sampto)
    record.d_signal = _slice_if_present(record.d_signal, sampfrom, sampto)

    record.sig_len = sampto - sampfrom

    record.comments.append('Preprocessing: slice of an original signal (%d..%d)'.format(sampto, sampfrom))
    return record
