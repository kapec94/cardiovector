from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import wfdb
from cycler import cycler
from mpl_toolkits.mplot3d import Axes3D

from cardiovector import transform
from ._lib import iterfy, getindices, get_analog


def _adjust_lim(lim, signal):
    if lim is not None:
        if len(lim) == 2:
            return lim
        else:
            raise ValueError('Expected limit to have two elements')

    return [np.min(signal), np.max(signal)]


_vcg_plot_types = ['3d', 'frontal', 'transverse', 'saggital']

PlotParams = namedtuple('PlotParams', 'x, y, z, xlim, ylim, zlim')


def _plot_3d(sp, p, plot_kw):
    ax = plt.subplot(sp, projection='3d')
    assert isinstance(ax, Axes3D)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_xlim(p.xlim)
    ax.set_ylim(p.ylim)
    ax.set_zlim(p.zlim)

    ax.plot(p.x, p.y, p.z, zdir='y', **plot_kw)


def _plot_2d(ax, xlabel, ylabel, x, y, xlim, ylim, plot_kw):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0, c='k')
    ax.axvline(0, c='k')
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.plot(x, y, **plot_kw)


def _plot_2d_frontal(subplot, p, plot_kw):
    ax = plt.subplot(subplot)
    _plot_2d(ax, 'X', 'Y', p.x, p.y, p.xlim, p.ylim, plot_kw)


def _plot_2d_transverse(subplot, p, plot_kw):
    ax = plt.subplot(subplot)
    _plot_2d(ax, 'X', 'Z', p.x, p.z, p.xlim, p.zlim, plot_kw)


def _plot_2d_saggital(subplot, p, plot_kw):
    ax = plt.subplot(subplot)
    _plot_2d(ax, 'Y', 'Z', p.y, p.z, p.ylim, p.zlim, plot_kw)


_plot_funcs = {
    '3d': _plot_3d,
    'frontal': _plot_2d_frontal,
    'saggital': _plot_2d_saggital,
    'transverse': _plot_2d_transverse
}


def _subplot_id(nrows, ncols, i):
    s = str.format('{0}{1}{2}', nrows, ncols, i)
    return int(s)


def _validate_plot_arg(plot):
    if plot == 'all':
        plot = _vcg_plot_types

    if isinstance(plot, str):
        plot = [plot]

    if not hasattr(plot, '__len__'):
        raise ValueError('Expected plot parameter to be array_like')

    for p in plot:
        if p not in _vcg_plot_types:
            raise ValueError('Unknown plot type ' + p)

    return plot


def _signame_hash(signame):
    return '#'.join(sorted(signame))


def _validate_signals_arg(record, signals):
    assert record is not None
    record = iterfy(record)

    assert len(record) > 0

    if signals is None:
        sigs = {_signame_hash(r.signame) for r in record}
        if len(sigs) > 1:
            raise ValueError('Records have different signal set. Don\'t know which to choose')

        return sigs.pop()

    return signals


def plotvcg(record, signals=None,
            plot='3d',
            xlim=None, ylim=None, zlim=None,
            figsize=4,
            plot_kw=None,
            fig_kw=None):
    """
    Prepare a plot of vectorcardiogram contained in an `wfdb.Record` instance.

    Parameters
    ----------
    record : wfdb.Record
        The record containing VCG signals to plot.
    signals : array_like, optional
        Array containing signal names corresponding to X, Y and Z lead of the VCG.
        If not specified and `record` only holds three signals function will plot them as X, Y, Z leads.
        Otherwise, if number of signals held in record isn't equal to three, an exception will be raised.
    plot : list of str or str, optional
        Specify what plots to include in the figure. May specify a single value or a list of them - following plots
        will be added as subplots to a figure.
        Currently accepted values are: '3d', 'frontal', 'saggital', 'transverse'. A special value 'all' will
        make the function plot all of the above.
    xlim, ylim, zlim : 2-tuple of int, optional
        Specify plot limits in X, Y, Z directions. If not set, they will be set accordingly to signals.
    figsize : int, optional
        Specify the size of each subplot. Subplots will be drawn as squares with side equals to `figsize` inches.
        Default is 4.
    plot_kw : dict, optional
        Keyword args to pass to plotting functions, namely ``pyplot.Axes.plot``.
        The default value is ``linewidth=0.75``.
    fig_kw : dict, optional
        Keyword args to pass to ``pyplot.figure`` function.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Figure containing all the drawn subplots.
    """

    if fig_kw is None:
        fig_kw = dict()
    if plot_kw is None:
        plot_kw = {'linewidth': 0.75}

    assert isinstance(record, wfdb.Record)

    signals = _validate_signals_arg(record, signals)
    plot = _validate_plot_arg(plot)

    if len(signals) != 3:
        raise ValueError('Expected signals to contain three elements')

    n = len(plot)
    if n == 4:
        ncols = 2
        nrows = 2
    else:
        ncols = 1
        nrows = n

    kx = 'x'
    ky = 'y'
    kz = 'z'
    k_ = [kx, ky, kz]

    raw_lims = {kx: xlim, ky: ylim, kz: zlim}

    data = dict()
    lims = dict()
    for k, s in zip(k_, signals):
        index = record.signame.index(s)
        psig = get_analog(record)[:, index].T

        data[k] = psig
        lims[k] = _adjust_lim(raw_lims[k], psig)

    fig = plt.figure(figsize=(ncols * figsize, nrows * figsize), **fig_kw)
    params = PlotParams(
        data[kx], data[ky], data[kz],
        lims[kx], lims[ky], lims[kz])

    for i, p in enumerate(plot):
        func = _plot_funcs[p]
        subplot = _subplot_id(nrows, ncols, i + 1)
        func(subplot, params, plot_kw)

    return fig


def plotrecs(record, signals=None, labels=None, sigtransform=None, fig_kw=None):
    """
    Plot multiple WFDB records.

    Parameters
    ----------
    record : (N,) array-like of wfdb.Record objects
        Iterable of records containing VCG signals.

    signals : (M,) array-like of str, optional
        Iterable of names of signals to plot. All the signals must be present in all records in `record` arg.

    labels : (N,) array-like of str, optional
        Labels to assign to plotted signals. By default this will be record names.

    sigtransform : function(X -> ndarray) -> ndarray, optional
        Function to call on every signal about to be plotted.

    fig_kw : dict, optional
        Additional positional arguments for creating a `matplotlib.pyplot.Figure` object.

    Returns
    -------
    matplotlib.pyplot.Figure
        Object containing plotted VCG signals.
    """

    # TODO: read units from record files

    record = iterfy(record)
    signals = _validate_signals_arg(record, signals)

    fig_kw = fig_kw or dict()
    sigtransform = sigtransform or transform.identity

    if labels is not None:
        assert (len(record) == len(labels))
    else:
        labels = [r.recordname for r in record]

    if len({r.fs for r in record}) != 1:
        raise ValueError('Records have different sampling rates. Will not continue!')

    fs = record[0].fs
    siglen = np.max([r.siglen for r in record])
    indices = [getindices(rec, signals) for rec in record]

    sigs = [get_analog(rec) for rec in record]
    vcgs = [sig[:, ind].T for sig, ind in zip(sigs, indices)]

    assert len({v.shape for v in vcgs}) == 1

    time = siglen / fs
    x = np.linspace(0, time, siglen)
    fig = plt.figure(**fig_kw)
    nsig = len(signals)

    axes = fig.subplots(nrows=nsig, sharex=True)
    if nsig == 1:
        axes = np.array([axes])

    for i, (ylabel, ax) in enumerate(zip(signals, axes.reshape(-1))):
        ax.set_prop_cycle(cycler('color', ['k', 'm', 'c', 'y']))

        for vcg, label in zip(vcgs, labels):
            v = sigtransform(vcg[i])
            ax.plot(x, v, label=label)

        ax.set_xlabel('time/s')
        ax.set_ylabel(ylabel + '/mV')

    plt.figlegend()
    return fig
