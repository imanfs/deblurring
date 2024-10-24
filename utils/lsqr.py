import numpy as np
import scipy as sp
from utils.utils import imblur
from utils.derivatives import applyDx, applyDy, applyDxTrans, applyDyTrans

### LSQR FUNCTIONS

def M_f(f,alpha,sigma,w,h,reg_type):
    Af = imblur(f,sigma,w,h)   
    if reg_type == "tikhonov":
        
        return np.concatenate((Af,np.sqrt(alpha)*f))
    
    elif reg_type == "gradient":
        Dx_f = applyDx(f.reshape(w,h)).flatten()
        Dy_f = applyDy(f.reshape(w,h)).flatten() 
        return np.concatenate((Af,np.sqrt(alpha)*Dx_f,np.sqrt(alpha)*Dy_f))


def MT_b(b,alpha,sigma,w,h,reg_type):
    if reg_type == "tikhonov":
        n = len(b)//2
        AT_b = imblur(b[:n],sigma,w,h)
        
        return  AT_b + np.sqrt(alpha)*b[n:]

    elif reg_type == "gradient":
        n = len(b)//3
        AT_b = imblur(b[:n],sigma,w,h) # because A = A_T, Ab == imblur(b)
        Dx_b = applyDxTrans(b[n:2*n].reshape(w,h)).flatten()
        Dy_b = applyDyTrans(b[2*n:].reshape(w,h)).flatten()
        
        return AT_b + np.sqrt(alpha)*Dx_b + np.sqrt(alpha)*Dy_b
    
def unblur_lsqr(g,alpha,sigma,w,h,reg_type,show=False):
    if reg_type == "tikhonov":
        b = np.vstack([np.reshape(g,(g.size,1)),np.zeros((g.size,1))])
        M,N = b.size,g.size

    elif reg_type == "gradient":
        b = np.vstack([np.reshape(g,(g.size,1)),np.zeros((g.size,1)),np.zeros((g.size,1))])
        M,N = b.size,g.size

    A = sp.sparse.linalg.LinearOperator((M,N),matvec = lambda f: M_f(f,alpha,sigma,w,h,reg_type), rmatvec = lambda b: MT_b(b,alpha,sigma,w,h,reg_type))
    lsqrOutput = sp.sparse.linalg.lsqr(A,b,show=show)
    f_lsqr = lsqrOutput[0]
    
    return f_lsqr.reshape(w,h)
