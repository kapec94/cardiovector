import wfdb
import numpy as np


def getindices(rec: wfdb.Record, sigs: list):
    return list(map(rec.sig_name.index, sigs))


def iterfy(iterable):
    if isinstance(iterable, str):
        iterable = [iterable]
    try:
        iter(iterable)
    except TypeError:
        iterable = [iterable]
    return iterable


def c_(a):
    if a is None:
        return None
    if isinstance(a, np.ndarray):
        return np.copy(a)
    if isinstance(a, list):
        return list(a)


def copy_record(r: wfdb.Record) -> wfdb.Record:
    assert isinstance(r, wfdb.Record)

    return wfdb.Record(c_(r.p_signal), c_(r.d_signal), c_(r.e_p_signal), c_(r.e_d_signal),
                       r.record_name, r.n_sig, r.fs, r.counter_freq, r.base_counter, r.sig_len,
                       r.base_time, r.base_date, r.file_name, r.fmt, r.samps_per_frame, r.skew, r.byte_offset,
                       r.adc_gain, r.baseline, r.units, r.adc_res, r.adc_zero, r.init_value, r.checksum, r.block_size,
                       r.sig_name, r.comments)


NONE_ATTRS = [
    'p_signal',
    'd_signal',
    'e_p_signal',
    'e_d_signal',
    'checksum'
]

SIMPLE_ATTRS = [
    'record_name',
    'fs',
    'counter_freq',
    'base_counter',
    'base_time',
    'base_date',
    'comments'
]

N_SIG_ATTRS = [
    'fmt',
    'file_name',
    'units',
    'samps_per_frame',
    'skew',
    'byte_offset',
    'adc_gain',
    'baseline',
    'units',
    'adc_res',
    'adc_zero',
    'init_value',
    'block_size'
]


def prepare_reshaped(r: wfdb.Record, n_sig: int, sig_len: int, **kwargs) -> wfdb.Record:
    def _or_default(attr, supplier):
        return attr, supplier() if attr not in kwargs else kwargs[attr]

    def _prepare_none_arg(r_, attr):
        if not hasattr(r_, attr):
            raise ValueError('Not such attr ' + attr)

        return _or_default(attr, lambda: None)

    def _prepare_simple_arg(r_, attr):
        if not hasattr(r_, attr):
            raise ValueError('Not such attr ' + attr)

        o = getattr(r_, attr)
        if hasattr(o, '__iter__') and not isinstance(o, str):
            o = c_(o)

        return _or_default(attr, lambda: o)

    def _prepare_list_arg(r_, attr):
        if not hasattr(r_, attr):
            raise ValueError('Not such attr ' + attr)

        def _eval():
            o = getattr(r_, attr)
            if not hasattr(o, '__iter__') or isinstance(o, str):
                raise ValueError('Non-iterable put as a list attr ' + attr)

            try:
                u = _get_uniq(o)
            except ValueError:
                raise ValueError('Could not get a unique value from ' + attr + '. Try specifying your own values')

            return [u] * n_sig

        return _or_default(attr, _eval)

    none_kwargs = dict(_prepare_none_arg(r, a) for a in NONE_ATTRS)
    simple_kwargs = dict(_prepare_simple_arg(r, a) for a in SIMPLE_ATTRS)
    list_kwargs = dict(_prepare_list_arg(r, a) for a in N_SIG_ATTRS)

    return wfdb.Record(**none_kwargs,
                       **simple_kwargs,
                       **list_kwargs,
                       n_sig=n_sig,
                       sig_len=sig_len)


def get_analog(r: wfdb.Record) -> np.ndarray:
    return _get_adac(r.p_signal, r.dac)


def get_digital(r: wfdb.Record) -> np.ndarray:
    return _get_adac(r.d_signal, r.adc)


def _get_adac(sig: np.ndarray, get):
    if sig is None or sig.shape == ():
        return get()
    else:
        return sig


def validate_adac(record: wfdb.Record) -> (int, int, int):
    fmt = _get_uniq(record.fmt)
    adcgain = _get_uniq(record.adc_gain)
    baseline = _get_uniq(record.baseline)
    adc_zero = _get_uniq(record.adc_zero)
    return fmt, adcgain, baseline, adc_zero


def _get_uniq(vals):
    s = set(vals)
    if len(s) != 1:
        raise ValueError('Could not get an unique value')

    return vals[0]
