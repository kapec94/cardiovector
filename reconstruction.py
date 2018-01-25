import numpy as np
from transform import identity
from _lib import getindices


def vcg_reconstruct_matrix(record, matrix, channels, nametransform=identity):
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

    nametransform : function(x -> string)
        function that will be used to create output record name from input record name.

    Returns
    -------
    wfdb.Record
        Record object containing reconstructed VCG signal.
    """

    from wfdb import Record

    signal_indices = [record.signame.index(sig) for sig in channels]
    input_vectors = record.p_signals[:, signal_indices]

    output_vectors = np.dot(matrix, input_vectors.T).A.T

    recordname = nametransform(record.recordname)
    return Record(recordname=recordname,
                  nsig=3, p_signals=output_vectors,
                  fs=record.fs, siglen=record.siglen,
                  signame=['vx', 'vy', 'vz'],
                  units=['mV', 'mv', 'mv'])


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

    return vcg_reconstruct_matrix(record, kors, kors_channels, nametransform=lambda x: x + '_kors')


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

    return vcg_reconstruct_matrix(record, idt, idt_channels, nametransform=lambda x: x + '_idt')


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
    from wfdb import Record

    sigs = dict(
        vx=['i', 'v5', 'v6'],
        vy=['ii', 'iii', 'avf'],
        vz=['v1', 'v2', 'v3'])

    signal_indices = [(out_sig, record.signame.index(in_sigs)) for out_sig, in_sigs in sigs.items()]
    input_vectors = record.p_signals[:, signal_indices]


vcg_methods = ['kors', 'idt', 'pca']


def vcg_reconstruct(record, method):
    assert method in vcg_methods

    if method == 'kors':
        return kors_vcg(record)
    if method == 'idt':
        return idt_vcg(record)
    if method == 'pca':
        return pca_vcg(record)
