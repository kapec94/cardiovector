def remove_mean(x):
    from numpy import mean
    return x - mean(x)


def identity(x): return x
