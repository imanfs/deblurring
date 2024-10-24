import scipy as sp
import numpy as np
from tqdm import tqdm
from utils.gmres import unblur_gmres
from utils.lsqr import unblur_lsqr

def imblur(f,sigma,w,h):
    """Blurs an image using a gaussian filter"""
    Af = sp.ndimage.gaussian_filter(f.reshape((w,h)),sigma)
    return Af.flatten()

class counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


def DP(alpha, g, sigma,theta,w,h,method,reg_type):
    if method == 'gmres':
        fi = unblur_gmres(g,alpha,sigma,w,h,reg_type)
    elif method == 'lsqr':
        fi = unblur_lsqr(g,alpha,sigma,w,h,reg_type)

    r = g - imblur(fi,sigma,w,h)
    r_norm = np.linalg.norm(r)
    n = g.size
    dp = (r_norm**2 / n) - theta**2
    return dp

def dp_array(alphas,g,sigma,theta,w,h,method,reg_type):
    dp = np.zeros(len(alphas))
    for i, alpha in tqdm(enumerate(alphas)):
        dp_vals = DP(alpha,g,sigma,theta,w,h,method,reg_type)
        dp[i] = dp_vals 

    return dp

def L_curve(g, alpha, sigma,theta,w,h,method,reg_type):
    if method == 'gmres':
        fi = unblur_gmres(g,alpha,sigma,w,h,reg_type)
    elif method == 'lsqr':
        fi = unblur_lsqr(g,alpha,sigma,w,h,reg_type)
    fi_norm = np.linalg.norm(fi)
    r = g - imblur(fi,sigma,w,h)
    r_norm = np.linalg.norm(r)
    return fi_norm,r_norm

def L_array(alphas,g,sigma,theta,w,h,method,reg_type):
    r_norms = np.zeros(len(alphas))
    fi_norms = np.zeros(len(alphas))
    for i, alpha in tqdm(enumerate(alphas)):
        fi_norm,r_norm = L_curve(g,alpha,sigma,theta,w,h,method,reg_type)
        fi_norms[i],r_norms[i] = fi_norm,r_norm
    return fi_norms,r_norms
