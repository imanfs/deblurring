import scipy as sp
from utils.utils import imblur
from utils.derivatives import laplacian


def ATA(f,alpha,sigma,w,h,reg_type,gamma=0):
    y = imblur(f,sigma,w,h)
    A_Ty = imblur(y,sigma,w,h)
    if reg_type == "tikhonov":
        z = A_Ty + alpha*f
    elif reg_type == "gradient":
        z = A_Ty + alpha*laplacian(f.reshape((w,h)),gamma)
    elif reg_type == "anisotropic":
        #z = A_Ty + alpha*(applyDxTrans(gamma*applyDx(f.reshape(w,h))).flatten()) +  alpha*(applyDyTrans(gamma*applyDy(f.reshape(w,h))).flatten())
        z = A_Ty + alpha*laplacian(f.reshape((w,h)),gamma)
    return z.reshape((-1,1))

def unblur_gmres(g,alpha,sigma,w,h,reg_type,callback=None,gamma=0):
    """
    Implements the GMRES solver for normal equations. 
    g is the blurred image to be reconstructed
    alpha is the regularisation parameter
    sigma is the blurring parameter passed to A (imblur(f))
    """
    M = g.size
    N = M
    ATg = imblur(g,sigma,w,h)
    A = sp.sparse.linalg.LinearOperator((M,N),
                                        matvec=lambda f: ATA(f,alpha,sigma,w,h,reg_type,gamma))
    gmresOutput = sp.sparse.linalg.gmres(A, ATg.flatten(),
                                         callback=callback,maxiter=2000) #gmres output
    f_alpha = gmresOutput[0]
    return f_alpha.reshape(w,h)
