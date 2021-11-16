%%cython

from cpython.array cimport array, clone
from libc.math cimport log, pow, sqrt
import numpy as np
cimport numpy as np

tFloat = np.double
ctypedef np.double_t dtype_t

cdef _ws2d(np.ndarray[dtype_t] y, double lmda, array[double] w):
    """Internal whittaker function for use in asymmetric smoothing.
    Args:
      y: time-series numpy array
      lmbda: lambda (s) value
      w: weights numpy array
    Returns:
        smoothed time-series array z
    """

    cdef array dbl_array_template = array('d', [])
    cdef int i, i1, i2, m, n
    cdef array z, d, c, e

    n = y.shape[0]
    m = n - 1

    z = clone(dbl_array_template, n, zero=False)
    d = clone(dbl_array_template, n, zero=False)
    c = clone(dbl_array_template, n, zero=False)
    e = clone(dbl_array_template, n, zero=False)

    d.data.as_doubles[0] = w.data.as_doubles[0] + lmda
    c.data.as_doubles[0] = (-2 * lmda) / d.data.as_doubles[0]
    e.data.as_doubles[0] = lmda /d.data.as_doubles[0]
    z.data.as_doubles[0] = w.data.as_doubles[0] * y[0]
    d.data.as_doubles[1] = w.data.as_doubles[1] + 5 * lmda - d.data.as_doubles[0] * (c.data.as_doubles[0] * c.data.as_doubles[0])
    c.data.as_doubles[1] = (-4 * lmda - d.data.as_doubles[0] * c.data.as_doubles[0] * e.data.as_doubles[0]) / d.data.as_doubles[1]
    e.data.as_doubles[1] =  lmda / d.data.as_doubles[1]
    z.data.as_doubles[1] = w.data.as_doubles[1] * y[1] - c.data.as_doubles[0] * z.data.as_doubles[0]
    for i in range(2, m-1):
        i1 = i - 1
        i2 = i - 2
        d.data.as_doubles[i]= w.data.as_doubles[i] + 6 *  lmda - (c.data.as_doubles[i1] * c.data.as_doubles[i1]) * d.data.as_doubles[i1] - (e.data.as_doubles[i2] * e.data.as_doubles[i2]) * d.data.as_doubles[i2]
        c.data.as_doubles[i] = (-4 *  lmda - d.data.as_doubles[i1] * c.data.as_doubles[i1] * e.data.as_doubles[i1])/ d.data.as_doubles[i]
        e.data.as_doubles[i] =  lmda / d.data.as_doubles[i]
        z.data.as_doubles[i] = w.data.as_doubles[i] * y[i] - c.data.as_doubles[i1] * z.data.as_doubles[i1] - e.data.as_doubles[i2] * z.data.as_doubles[i2]
    i1 = m - 2
    i2 = m - 3
    d.data.as_doubles[m - 1] = w.data.as_doubles[m - 1] + 5 *  lmda - (c.data.as_doubles[i1] * c.data.as_doubles[i1]) * d.data.as_doubles[i1] - (e.data.as_doubles[i2] * e.data.as_doubles[i2]) * d.data.as_doubles[i2]
    c.data.as_doubles[m - 1] = (-2 *  lmda - d.data.as_doubles[i1] * c.data.as_doubles[i1] * e.data.as_doubles[i1]) / d.data.as_doubles[m - 1]
    z.data.as_doubles[m - 1] = w.data.as_doubles[m - 1] * y[m - 1] - c.data.as_doubles[i1] * z.data.as_doubles[i1] - e.data.as_doubles[i2] * z.data.as_doubles[i2]
    i1 = m - 1
    i2 = m - 2
    d.data.as_doubles[m] = w.data.as_doubles[m] +  lmda - (c.data.as_doubles[i1] * c.data.as_doubles[i1]) * d.data.as_doubles[i1] - (e.data.as_doubles[i2] * e.data.as_doubles[i2]) * d.data.as_doubles[i2]
    z.data.as_doubles[m] = (w.data.as_doubles[m] * y[m] - c.data.as_doubles[i1] * z.data.as_doubles[i1] - e.data.as_doubles[i2] * z.data.as_doubles[i2]) / d.data.as_doubles[m]
    z.data.as_doubles[m - 1] = z.data.as_doubles[m - 1] / d.data.as_doubles[m - 1] - c.data.as_doubles[m - 1] * z.data.as_doubles[m]
    for i in range(m-2, -1, -1):
        z.data.as_doubles[i] = z.data.as_doubles[i] / d.data.as_doubles[i] - c.data.as_doubles[i] * z.data.as_doubles[i + 1] - e.data.as_doubles[i] * z.data.as_doubles[i + 2]
    return z

cpdef ws2doptvp(np.ndarray[dtype_t] y, np.ndarray[dtype_t] w, array[double] llas, double p):
    """Whittaker smoother with asymmetric V-curve optimization of lambda (S).
    Args:
        y: time-series numpy array
        w: weights numpy array
        llas: array with lambda values to iterate (S-range)
        p: "Envelope" value
    Returns:
        Smoothed time-series array z and optimized lambda (S) value lopt
    """
    cdef array template = array('d', [])
    cdef array fits, pens, diff1, lamids, v, z
    cdef int m, m1, m2, nl, nl1, lix, i, j, k
    cdef double w_tmp, y_tmp, z_tmp, z2, llastep, fit1, fit2, pen1, pen2, l, l1, l2, vmin, lopt, p1

    m = y.shape[0]
    m1 = m - 1
    m2 = m - 2
    nl = len(llas)
    nl1 = nl - 1
    i = 0
    k = 0
    j = 0
    p1 = 1-p

    template = array('d', [])
    fits = clone(template, nl, True)
    pens = clone(template, nl, True)
    z = clone(template, m, True)
    znew = clone(template, m, True)
    diff1 = clone(template, m1, True)
    lamids = clone(template, nl1, False)
    v = clone(template, nl1, False)
    wa = clone(template, m, False)
    ww = clone(template, m, False)

    # Compute v-curve
    for lix in range(nl):
        l = pow(10,llas.data.as_doubles[lix])

        for i in range(10):
          for j in range(m):
            y_tmp = y[j]
            z_tmp = z.data.as_doubles[j]
            if y_tmp > z_tmp:
              wa.data.as_doubles[j] = p
            else:
              wa.data.as_doubles[j] = p1
            ww.data.as_doubles[j] = w[j] * wa.data.as_doubles[j]

          znew[0:m] = _ws2d(y, l, ww)
          z_tmp = 0.0
          j = 0
          for j in range(m):
            z_tmp += abs(znew.data.as_doubles[j] - z.data.as_doubles[j])

          if z_tmp == 0.0:
            break

          z[0:m]= znew[0:m]

        for i in range(m):
            w_tmp = w[i]
            y_tmp = y[i]
            z_tmp = z.data.as_doubles[i]
            fits.data.as_doubles[lix] += pow(w_tmp * (y_tmp - z_tmp),2)
        fits.data.as_doubles[lix] = log(fits.data.as_doubles[lix])

        for i in range(m1):
            z_tmp = z.data.as_doubles[i]
            z2 = z.data.as_doubles[i+1]
            diff1.data.as_doubles[i] = z2 - z_tmp
        for i in range(m2):
            z_tmp = diff1.data.as_doubles[i]
            z2 = diff1.data.as_doubles[i+1]
            pens.data.as_doubles[lix] += pow(z2 - z_tmp,2)
        pens.data.as_doubles[lix] = log(pens.data.as_doubles[lix])

    # Construct v-curve
    llastep = llas[1] - llas[0]

    for i in range(nl1):
        l1 = llas.data.as_doubles[i]
        l2 = llas.data.as_doubles[i+1]
        fit1 = fits.data.as_doubles[i]
        fit2 = fits.data.as_doubles[i+1]
        pen1 = pens.data.as_doubles[i]
        pen2 = pens.data.as_doubles[i+1]
        v.data.as_doubles[i] = sqrt(pow(fit2 - fit1,2) + pow(pen2 - pen1,2)) / (log(10) * llastep)
        lamids.data.as_doubles[i] = (l1+l2) / 2

    vmin = v.data.as_doubles[k]
    for i in range(1, nl1):
        if v.data.as_doubles[i] < vmin:
            vmin = v.data.as_doubles[i]
            k = i

    lopt = pow(10, lamids.data.as_doubles[k])

    del z
    z = clone(template, m, True)

    for i in range(10):
      for j in range(m):
        y_tmp = y[j]
        z_tmp = z.data.as_doubles[j]

        if y_tmp > z_tmp:
          wa.data.as_doubles[j] = p
        else:
          wa.data.as_doubles[j] = p1
        ww.data.as_doubles[j] = w[j] * wa.data.as_doubles[j]

      znew[0:m] = _ws2d(y, lopt, ww)
      z_tmp = 0.0
      j = 0
      for j in range(m):
        z_tmp += abs(znew.data.as_doubles[j] - z.data.as_doubles[j])

      if z_tmp == 0.0:
        break

      z[0:m]= znew[0:m]

    z[0:m] = _ws2d(y, lopt, ww)
    
    return z, lopt, v, lamids
