import numpy as np
import wfdb
from sklearn.decomposition import PCA

from ._lib import get_digital, validate_adac


class ReconstructionBase:
    def reconstruct(self, record: wfdb.Record):
        from wfdb import Record

        fmt, adcgain, baseline = validate_adac(record)

        channels = self.channels()
        signal_indices = [record.sig_name.index(sig) for sig in channels]

        digital = get_digital(record)

        input_vectors_raw = digital[:, signal_indices]
        input_vectors = np.array([self._preprocess(c, s) for c, s in zip(channels, input_vectors_raw.T)]).T

        output_vectors = self._reconstruct(input_vectors)
        nsig = output_vectors.shape[1]

        return Record(record_name=record.record_name,
                      n_sig=nsig, d_signal=output_vectors,
                      fs=record.fs, sig_len=record.sig_len,
                      fmt=[fmt] * nsig,
                      adc_gain=[adcgain] * nsig,
                      baseline=[baseline] * nsig,
                      sig_name=['vx', 'vy', 'vz'],
                      units=['mV', 'mv', 'mv'])

    def channels(self) -> list:
        raise NotImplementedError()

    def _preprocess(self, channel_name: str, signal: np.ndarray):
        return signal

    def _reconstruct(self, input_vectors: np.ndarray):
        raise NotImplementedError()


class MatrixReconstruction(ReconstructionBase):
    def __init__(self, channels, matrix):
        if not hasattr(channels, '__iter__') or isinstance(channels, str):
            raise ValueError('Channels must be non-empty array-like')

        if len(channels) != matrix.shape[1]:
            raise ValueError('Channel count must equal to MxN matrix M-dimension')

        self._channels = channels
        self._matrix = matrix

    def matrix(self):
        return self._matrix

    def channels(self):
        return self._channels

    def _reconstruct(self, input_vectors):
        return np.dot(self._matrix, input_vectors.T).A.T


pca_channels = ['i', 'v5', 'v6', 'ii', 'iii', 'avf', 'v1', 'v2', 'v3']


class PcaReconstruction(ReconstructionBase):
    def channels(self):
        return pca_channels

    def _reconstruct(self, input_vectors):
        assert len(pca_channels) == 9
        assert input_vectors.shape[1] == len(pca_channels)

        sigs = {
            'vx': input_vectors[:, 0:3],
            'vy': input_vectors[:, 3:6],
            'vz': input_vectors[:, 6:9]
        }
        output_vectors = list()

        for sig in sigs.values():
            output = self._sig_pca(sig)
            output_vectors.append(output)

        return np.array(output_vectors).T

    @staticmethod
    def _sig_pca(signals):
        pca = PCA(n_components=1)
        out = pca.fit_transform(signals)

        return -out[:, 0]


def vcg_reconstruct_matrix(record, matrix, channels):
    """
    Reconstruct VCG from 12-lead ECG using a matrix reconstruction algorithm.

    Parameters
    ----------
    record : wfdb.Record
        Record object containing the 12-lead ECG signal.

    matrix : (N,M) matrix
        NxM matrix that will be used to transform ECG signal.

    channels : (M,) array-like
        channel names to retrieve signals from the record.


    Returns
    -------
    wfdb.Record
        Record object containing reconstructed VCG signal.
    """

    reconstruction = MatrixReconstruction(channels, matrix)
    return reconstruction.reconstruct(record)


kors_channels = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'i', 'ii']
kors = np.matrix([[-0.13, 0.05, -0.01, 0.14, 0.06, 0.54, 0.38, -0.07],  # x
                  [0.06, -0.02, -0.05, 0.06, -0.17, 0.13, -0.07, 0.93],  # y
                  [-0.43, -0.06, -0.14, -0.20, -0.11, 0.31, 0.11, -0.23]])  # z


def kors_vcg(record):
    """
    Reconstruct VCG from 12-lead ECG using Kors regression matrix algorithm.

    Parameters
    ----------
    record : wfdb.Record
        Record object containing the 12-lead ECG signal.

    Returns
    -------
    wfdb.Record
        Record object containing reconstructed VCG signal.
    """

    return vcg_reconstruct_matrix(record, kors, kors_channels)


idt_channels = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'i', 'ii']
idt = np.matrix([[-0.17, -0.07, 0.12, 0.23, 0.24, 0.19, 0.16, -0.01],  # x
                 [0.06, -0.02, -0.11, -0.02, 0.04, 0.05, -0.23, 0.89],  # y
                 [-0.23, -0.31, -0.25, -0.06, 0.05, 0.11, 0.02, 0.10]])  # z


def idt_vcg(record):
    """
    Reconstruct VCG from 12-lead ECG using Inverse Dower Transform matrix algorithm.

    Parameters
    ----------
    record : wfdb.Record
        Record object containing the 12-lead ECG signal.

    Returns
    -------
    wfdb.Record
        Record object containing reconstructed VCG signal.
    """

    return vcg_reconstruct_matrix(record, idt, idt_channels)


def pca_vcg(record):
    """
    Reconstruct VCG from 12-lead ECG using PCA-based reconstruction algorithm.

    Parameters
    ----------
    record : wfdb.Record
        Record object containing the 12-lead ECG signal.

    Returns
    -------
    wfdb.Record
        Record object containing reconstructed VCG signal.
    """

    reconstruction = PcaReconstruction()
    return reconstruction.reconstruct(record)


vcg_methods = ['kors', 'idt', 'pca']


def vcg_reconstruct(record, method):
    assert method in vcg_methods

    if method == 'kors':
        return kors_vcg(record)
    if method == 'idt':
        return idt_vcg(record)
    if method == 'pca':
        return pca_vcg(record)
