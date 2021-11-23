import numba
from numba import jit#, float64, int64
# from numpy import zeros, isnan

def input_stats(rstack_array):
    '''
    ['rs_min', 'rs_max', 'rs_range', 'rs_mean',
     'rs_std', 'rs_med', 'rs_uq', 'rs_lq',
     'rs_iqr', 'rs_snr', 'rs_snr2', 'rs_lag1']
    '''
    rs_arr = copy.copy(rstack_array)

    rs_min = np.nanmin(rs_arr, axis=0)
    rs_max = np.nanmax(rs_arr, axis=0)
    rs_range = rs_max - rs_min

    rs_mean = np.nanmean(rs_arr, axis=0)
    rs_std = np.nanstd(rs_arr, axis=0)

    rs_med = np.nanmedian(rs_arr, axis=0)
    rs_uq = np.nanquantile(rs_arr, 0.75, axis=0)
    rs_lq = np.nanquantile(rs_arr, 0.25, axis=0)
    rs_iqr = rs_uq - rs_lq

    rs_snr = rs_mean/rs_std
    rs_snr2 = rs_snr**2
    
    rs_lag1 = np.apply_along_axis(lag1corr, 0, rs_arr)

    stats_arr = [rs_min, rs_max, rs_range, rs_mean,
                 rs_std, rs_med, rs_uq, rs_lq,
                 rs_iqr, rs_snr, rs_snr2, rs_lag1]
    
    stats_arr = np.array(stats_arr)

    return stats_arr

@jit(nopython=True)
def lag1corr(data_series, nodata=-3000):
    """Calculates Lag-1 autocorrelation.

    Adapted from https://stackoverflow.com/a/29194624/5997555

    Args:
        xx: first data series
        yy: second data series
        nodata: no-data value (will be exluded from calulation)

    Returns:
        Lag-1 autocorrelation value
    """
    xx = data_series[:-1]
    yy = data_series[1:]
    N = xx.shape[0]


    #   ((X - X.mean())*(Y - Y.mean())).mean()
    # =  Sxy*nxy - Sx_*Sy_
    Sx_ = numba.int64(0)  # Sum(Xi)    when Xi is valid and Yi is valid
    Sy_ = numba.int64(0)  # Sum(Yi)    when Xi is valid and Yi is valid
    Sxy = numba.int64(0)  # Sum(Xi*Yi) when Xi and Yi are both valid
    nxy = numba.int64(0)  # number of valid Xi,Yi tuples (both non-nodata)

    # var(X) = Sxx*nx - Sx*Sx
    Sx = numba.int64(0)  # Sum(Xi)    when Xi is valid
    Sxx = numba.int64(0)  # Sum(Xi*Xi) when Xi is valid
    nx = numba.int64(0)  # number of valid Xi

    # var(Y) = Syy*ny - Sy*Sy
    Sy = numba.int64(0)  # Sum(Yi)    when Yi is valid
    Syy = numba.int64(0)  # Sum(Yi*Yi) when Yi is valid
    ny = numba.int64(0)  # number of valid Yi

    for i in range(N):
        x = xx[i]
        y = yy[i]

        if x != nodata:
            Sx += x
            Sxx += x * x
            nx += 1

        if y != nodata:
            Sy += y
            Syy += y * y
            ny += 1

        if x != nodata and y != nodata:
            Sx_ += x
            Sy_ += y
            Sxy += x * y
            nxy += 1

    result = numba.float64(0.0)  # or should this be nan?
    if nxy == 0:
        return result

    A = nxy * numba.float64(Sxy) - numba.float64(Sx_) * numba.float64(Sy_)

    # var(X[np.isfinite(X)]) Vairance of X excluding missing values
    var_X = nx * numba.float64(Sxx) - numba.float64(Sx) * numba.float64(Sx)
    var_Y = ny * numba.float64(Syy) - numba.float64(Sy) * numba.float64(Sy)

    # var(X) where missing values were replaced with mean,
    #   i.e. X[X==nodata] = mean(X[X!=nodata])
    var_X = var_X * nx / N
    var_Y = var_Y * ny / N

    if var_X < 1e-8 or var_Y < 1e-8:
        return result

    result = A * (var_X ** -0.5) * (var_Y ** -0.5)
    return result
