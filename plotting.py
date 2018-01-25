def plot_vcg_3d(record, signals=None):
    import wfdb

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    assert isinstance(record, wfdb.Record)

    if signals is None:
        if record.nsig != 3:
            raise ValueError('No explicit signal names provided; don\'t know which signals to choose')

        signals = record.signame

    if len(signals) != 3:
        raise ValueError('Expected signals to contain three elements')

    data = dict((s, record.p_signals[s]) for s in signals)

