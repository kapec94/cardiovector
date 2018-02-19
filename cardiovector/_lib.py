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


def copy_record(r: wfdb.Record):
    assert isinstance(r, wfdb.Record)

    def c_(a): return None if a is None else np.array(a, copy=True)
    return wfdb.Record(c_(r.p_signal), c_(r.d_signal), c_(r.e_p_signal), c_(r.e_d_signal),
                       r.record_name, r.n_sig, r.fs, r.counter_freq, r.base_counter, r.sig_len,
                       r.base_time, r.base_date, r.file_name, r.fmt, r.samps_per_frame, r.skew, r.byte_offset,
                       r.adc_gain, r.baseline, r.units, r.adc_res, r.adc_zero, r.init_value, r.checksum, r.block_size,
                       r.sig_name, r.comments)


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
    return fmt, adcgain, baseline


def _get_uniq(vals):
    s = set(vals)
    if len(s) != 1:
        raise ValueError('Could not get an unique value')

    return vals[0]
