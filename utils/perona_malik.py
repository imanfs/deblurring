
import numpy as np
import scipy as sp
from utils.derivatives import applyDx, applyDy

def perona_malik(f):
    assert f.ndim > 1
    n = f.shape[0]
    Dx2 = np.square(applyDx(f))
    Dy2 = np.square(applyDy(f))
    T = np.sqrt(Dx2 + Dy2).max()
    gamma = np.exp(-np.sqrt(Dx2 + Dy2)/T)
    gamma = sp.sparse.spdiags(gamma.ravel(),diags=0,m=n,n=n).toarray()
    return gamma
