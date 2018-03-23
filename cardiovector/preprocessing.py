import deprecation as deprecation
import pywt
import wfdb
import numpy as np

from ._lib import copy_record


class BaseProcessor:
    def process(self, record: wfdb.Record):
        if not isinstance(record, wfdb.Record):
            raise ValueError('Not a wfdb.Record. For single signals, use _s version of that function')

        if record.d_signal is None:
            raise ValueError('Record should have digital signals')

        record = copy_record(record)
        for i in range(record.n_sig):
            signal = record.d_signal[:, i]
            new_signal = self.process_s(signal)
            if new_signal.shape != signal.shape:
                raise ValueError('process_s method must not reshape the signal')

            record.d_signal[:, i] = new_signal

        comment = self._comment()
        if comment is not None:
            record.comments.append(comment)

        return record

    def process_s(self, signal: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _comment(self) -> str:
        return None


class BWRemover(BaseProcessor):
    _wavelets = ['sym10', 'db2']

    def __init__(self, wavelet=None, min_level=10):
        self.wavelet = wavelet
        self.min_level = min_level

    @staticmethod
    def choose_wavelet(signal_length, preference=None, min_level=10):
        """
        Make a decision on to which wavelet should be used for BW filtering. If no decision can be made, due to not
        enough data samples, `fallback` will be returned if available.

        Currently there are two wavelets considered, both at 10th level of decomposition: symmlet-10 and daubechies-2.
        Both were proven effective for BW filtering (@Bunluechokchai2010, @Maheshwari16).

        Parameters
        ----------
        signal_length : int
            Number of samples in the signal to process.
        preference : str, optional
            Wavelet preferred by the user. If not specified, one of the built-in will be used.
        min_level : int, optional
            Minimum decomposition level required by the user. Defaults to 10.

        Returns
        -------
        wavelet_level : (wavelet : str, level : int)
            Tuple containing name and desired level of selected wavelet to use for filtering.
        """

        if preference is not None:
            max_level = pywt.dwt_max_level(signal_length, pywt.Wavelet(preference))
            if max_level > min_level:
                raise ValueError('Wavelet ' + preference + ' could not be used on a level ' + min_level
                                 + '. Not enough data!')

            return preference, max_level

        for wavelet in BWRemover._wavelets:
            max_level = pywt.dwt_max_level(signal_length, pywt.Wavelet(wavelet))
            if max_level >= min_level:
                return wavelet, max_level

        raise ValueError('Could not find a wavelet that would be good enough. Sample is too small!')

    def _comment(self) -> str:
        return 'Filtering: removed base wandering'

    def process_s(self, signal: np.ndarray) -> np.ndarray:
        signal_length = len(signal)
        wavelet, max_level = BWRemover.choose_wavelet(signal_length=signal_length,
                                                      preference=self.wavelet,
                                                      min_level=self.min_level)

        ca_last_orig, *cDn = pywt.wavedecn(signal, wavelet)
        ca_last = np.zeros(ca_last_orig.shape)

        result = pywt.waverecn([ca_last, *cDn], wavelet)
        so = result.shape[0]
        si = signal.shape[0]
        if so > si:
            n = si - so
            result = result[:n]

        return result


class NoiseRemover(BaseProcessor):
    def __init__(self, wavelet=None, min_level=9, threshold=3):
        assert threshold <= min_level

        self.wavelet = wavelet
        self.min_level = min_level
        self.threshold = threshold

    def process_s(self, signal: np.ndarray) -> np.ndarray:
        wavelet, level = NoiseRemover._choose_wavelet(len(signal), self.wavelet, self.min_level)

        coeffs = pywt.swt(signal, wavelet, level=level)
        for i in range(self.threshold):
            NoiseRemover._zero_coeff(coeffs, -(i + 1))

        out = pywt.iswt(coeffs, 'sym8')
        return out

    def _comment(self) -> str:
        return 'Filtering: removed noise'

    @staticmethod
    def _zero_coeff(where, which):
        ca, cd = where[which]
        where[which] = (np.zeros(ca.shape), np.zeros(cd.shape))

    @staticmethod
    def _choose_wavelet(signal_length, wavelet, min_level, fallback='sym8'):
        ws = []
        if wavelet is not None:
            ws.append(wavelet)
        ws.append(fallback)

        for w in ws:
            max_level = pywt.dwt_max_level(signal_length, pywt.Wavelet(w))
            if max_level > min_level:
                return w, min_level

        raise ValueError('Could not choose the wavelet')


@deprecation.deprecated(details='use BWRemover.choose_wavelet instead',
                        deprecated_in='0.1.2', removed_in='0.2.0')
def choose_wavelet(signal_length, fallback=None):
    return BWRemover.choose_wavelet(signal_length=signal_length, preference=fallback[0], min_level=fallback[1])


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
    wavelet_level = wavelet_level or (None, 10)
    return BWRemover(wavelet=wavelet_level[0], min_level=wavelet_level[1]).process_s(signal)


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
    wavelet_level = wavelet_level or (None, 10)
    return BWRemover(wavelet=wavelet_level[0], min_level=wavelet_level[1]).process(record)


def remove_noise(record: wfdb.Record, wavelet=None, threshold=3, min_level=9) -> wfdb.Record:
    """
    Remove noise from ECG/VCG signal using Translation-Invariant Wavelet Transform.

    Parameters
    ----------
    record : wfdb.Record
        Record to filter
    wavelet : str, optional
        Which wavelet to use. If not specified, a built-in wavelet will be used.
    threshold : int, optional
        How many decomposition levels to clear. Defaults to 3.
    min_level : int, optional
        Minimum decomposition level. Defaults to 10.

    Returns
    -------
    wfdb.Record
        Filtered record.
    """
    return NoiseRemover(wavelet=wavelet,
                        min_level=min_level,
                        threshold=threshold).process(record)


def remove_noise_s(signal, wavelet=None, threshold=3, min_level=9):
    """
    Remove noise from ECG/VCG signal using Translation-Invariant Wavelet Transform.

    Parameters
    ----------
    signal : (N,) array_like
        ECG signal to filter.
    wavelet : str, optional
        Which wavelet to use. If not specified, a built-in wavelet will be used.
    threshold : int, optional
        How many decomposition levels to clear. Defaults to 3.
    min_level : int, optional
        Minimum decomposition level. Defaults to 10.

    Returns
    -------
    out_signal : (N,) numpy.array
        Filtered signal.
    """
    return NoiseRemover(wavelet=wavelet,
                        min_level=min_level,
                        threshold=threshold).process_s(signal)


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
