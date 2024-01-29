from datetime import datetime
import random
import pickle

def getTimeStamp():
    # timestamp
    ts = datetime.today()
    timeStamp = ts.strftime('%Y%m%d%H%M%S')

    # randomID
    source_str = 'abcdefghijklmnopqrstuvwxyz'
    randomID = ''.join([random.choice(source_str) for x in range(10)])
    timeStamp += '-'+randomID
    return timeStamp

import numba as nb
import numpy as np
from numba import jit, f8, i8, u1, b1
from scipy.special import logsumexp
# only for 1dim array

@jit(nopython=True)
def _logsumexp(a):
    if a.ndim > 1:
        print('alogsumexp takes only 1-dim')
    axis = 0
    #a_max = np.amax(a, axis=axis, keepdims=True)
    #print(a_max, a_max.ndim)
    a_max = np.max(a)
    #print(np.max(a, axis=0))

    if not np.isfinite(a_max):
        a_max = 0

    tmp = np.exp(a - a_max)

    s = np.sum(tmp)
    out = np.log(s)

    out += a_max

    return out

