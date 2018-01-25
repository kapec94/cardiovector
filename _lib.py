from transform import identity

def getindices(rec, sigs):
    return list(map(rec.signame.index, sigs))

def plot_vcgs(records, labels, transform=identity, figsize=None):
    """
    Plot VCG signals (vx, vy, vz) from multiple records, in order to compare them.

    Parameters
    ----------
    records : (N,) array-like of wfdb.Record objects
        Iterable of records containing VCG signals.

    labels : (N,) array-like
        Iterable of signal labels.

    transform : function(X -> ndarray) -> ndarray, optional
        Function to call on every signal about to be plotted.

    figsize : (2,) array-like, optional
        Figure size as configure in matplotlib.

    Returns
    -------
    matplotlib.pyplot.Figure
        Plotted Figure object.
    """

    from cycler import cycler
    from numpy import linspace
    import matplotlib.pyplot as plt

    assert (len(records) == len(labels))
    assert (len({r.fs for r in records}) == 1)
    assert (len({r.siglen for r in records}) == 1)

    signames = ['vx', 'vy', 'vz']

    fs = records[0].fs
    siglen = records[0].siglen

    def _getindices(rec, sigs):
        return list(map(rec.signame.index, sigs))

    indices = [_getindices(rec, signames) for rec in records]
    vcgs = [rec.p_signals[:, ind].T for rec, ind in zip(records, indices)]

    assert (len({v.shape for v in vcgs}) == 1)

    x = linspace(0, siglen / fs, siglen)

    fig = plt.figure(figsize=(figsize or (10, 8)))
    axes = fig.subplots(nrows=3, sharex=True)

    for i, (ylabel, ax) in enumerate(zip(signames, axes.reshape(-1))):
        ax.set_prop_cycle(cycler('color', ['k', 'm', 'c', 'y']))
        for vcg, label in zip(vcgs, labels):
            v = transform(vcg[i])
            ax.plot(x, v, label=label)
        ax.set_xlabel('time/s')
        ax.set_ylabel(ylabel + '/V')

    plt.figlegend()
    return fig
