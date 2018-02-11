import wfdb
import numpy as np


def getindices(rec, sigs):
    return list(map(rec.signame.index, sigs))


def iterfy(iterable):
    if isinstance(iterable, str):
        iterable = [iterable]
    try:
        iter(iterable)
    except TypeError:
        iterable = [iterable]
    return iterable


def copy_record(r):
    assert isinstance(r, wfdb.Record)

    def c_(a): return None if a is None else np.array(a, copy=True)
    return wfdb.Record(c_(r.p_signals), c_(r.d_signals), c_(r.e_p_signals), c_(r.e_d_signals),
                       r.recordname, r.nsig, r.fs, r.counterfreq, r.basecounter, r.siglen,
                       r.basetime, r.basedate, r.filename, r.fmt, r.sampsperframe, r.skew, r.byteoffset,
                       r.adcgain, r.baseline, r.units, r.adcres, r.adczero, r.initvalue, r.checksum, r.blocksize,
                       r.signame, r.comments)


def get_analog(r):
    return _get_adac(r.p_signals, r.dac)


def get_digital(r):
    return _get_adac(r.d_signals, r.adc)


def _get_adac(sig, get):
    if sig is None or sig.shape == ():
        return get()
    else:
        return sig


def validate_adac(record):
    fmt = _get_uniq(record.fmt)
    adcgain = _get_uniq(record.adcgain)
    baseline = _get_uniq(record.baseline)
    return fmt, adcgain, baseline


def _get_uniq(vals):
    s = set(vals)
    if len(s) != 1:
        raise ValueError('Could not get an unique value')

    return vals[0]
