import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from utils.utils import imblur, counter, dp_array, DP, L_curve, L_array
from utils.gmres import unblur_gmres
from utils.lsqr import unblur_lsqr
from utils.derivatives import applyDx, applyDy
from utils.plotting import format_plot, plot_deblur
from utils.perona_malik import perona_malik

ftrue = plt.imread('house.png')
w,h = ftrue.shape
sigma = 2  # gaussian blur
theta = 0.05  # noise
alpha = 0.05  # regularisation

g = imblur(ftrue,sigma,w,h)
g = g + theta*np.random.randn(g.size)

fig,ax = plt.subplots(1,2)
ax[0].imshow(ftrue,cmap="gray")
ax[1].imshow(g.reshape(w,h),cmap="gray")
format_plot('1a')


regulariser = "tikhonov"

f_alpha = unblur_gmres(g,alpha,sigma,w,h,regulariser,callback=counter())
plot_deblur(f_alpha,ftrue,g,sigma,'1c')

f_alpha = unblur_lsqr(g,alpha,sigma,w,h,regulariser,show=True)
plot_deblur(f_alpha,ftrue,g,sigma,'1d')

method = "gmres"
alphas = np.linspace(1e-2,1e-1,20)
dp = dp_array(alphas,g, sigma,theta,w,h,method,regulariser)

alpha_idx = min(range(len(dp)), key=lambda i: abs(dp[i]))
alpha_binarysearch = alphas[alpha_idx]
plt.scatter(alpha_binarysearch,dp[alpha_idx],c='r',marker="*",zorder=5,label="Binary search")

solution = sp.optimize.root(DP, 0.5e-2,args=(g,sigma,theta,w,h,method,regulariser))
if solution.success: 
    opt_alpha = solution.x[0]

opt_alpha_GMRES = opt_alpha
plt.plot(alphas,dp,'.-',lw=0.7)
plt.scatter(alpha_binarysearch,dp[alpha_idx],c='r',marker="*",zorder=5,label="Binary search")
plt.scatter(opt_alpha,DP(opt_alpha,g,sigma,theta,w,h,method,regulariser),c="g",marker="x",zorder=10,label="SciPy root")
ax = plt.gca()
ax.set_xscale('log')
plt.xlabel(r"$\alpha")
plt.ylabel("DP")
plt.legend()
format_plot('2i',show_axis=True)


method = "lsqr"
alphas = np.linspace(1e-2,1e-1,20)
dp = dp_array(alphas,g, sigma,theta,w,h,method,regulariser)

alpha_idx = min(range(len(dp)), key=lambda i: abs(dp[i]))
alpha_binarysearch = alphas[alpha_idx]

solution = sp.optimize.root(DP, 0.5e-2,args=(g,sigma,theta,w,h,method,regulariser))
if solution.success: 
    opt_alpha = solution.x[0]
opt_alpha_LSQR = opt_alpha

opt_alpha_LSQR, opt_alpha_GMRES

fi_norm_dp, r_norm_dp = L_curve(g, opt_alpha, sigma,theta,w,h,method,regulariser)

alphas = 10.0**np.arange(-3,0,0.25)
fi_norms,r_norms = L_array(alphas,g,sigma,theta,w,h,method,regulariser)
fig,ax = plt.subplots(1,2,figsize=(12,6))
ax[0].loglog(fi_norms,r_norms,'.-',lw=0.7)
ax[0].scatter(fi_norm_dp,r_norm_dp,c='g',marker='x',label=r"$\alpha$ DP")

alphas = 10.0**np.linspace(-3,-1,12)
fi_norms,r_norms = L_array(alphas,g,sigma,theta,w,h,method,regulariser)
ax[1].loglog(fi_norms,r_norms,'.-',lw=0.7,label = "L-curve residuals")
ax[1].scatter(fi_norm_dp,r_norm_dp,c='g',marker='x',label=r"$\alpha$ DP")

plt.legend()
fig.supxlabel(f'$||f||$')
fig.supylabel(f'$||r||$')
fig.suptitle("L-curve for GMRES solver with Tikhonov regularisation")
format_plot('2ii')

f_alpha = unblur_gmres(g,opt_alpha,sigma,w,h,"tikhonov")
plot_deblur(f_alpha,ftrue,g,sigma,"2iii")

regulariser = "gradient"
fig,ax = plt.subplots(1,2)
ax[0].imshow(applyDx(ftrue.reshape(h,w)),cmap = "gray")
ax[1].imshow(applyDy(ftrue.reshape(h,w)),cmap = "gray")
format_plot('3a')

f_alpha = unblur_gmres(g,alpha,sigma,w,h,regulariser,callback=counter())
plot_deblur(f_alpha,ftrue,g,sigma,'3bi')

f_alpha = unblur_lsqr(g,alpha,sigma,w,h,regulariser,show=True)
plot_deblur(f_alpha,ftrue,g,sigma,'3bii')

method = "gmres"
alphas = np.linspace(1e-2,1e-1,20)
dp = dp_array(alphas,g, sigma,theta,w,h,method,regulariser)

alpha_idx = min(range(len(dp)), key=lambda i: abs(dp[i]))
alpha_binarysearch = alphas[alpha_idx]

solution = sp.optimize.root(DP, 0.5e-2,args=(g,sigma,theta,w,h,method,regulariser))
if solution.success: 
    opt_alpha = solution.x[0]

opt_alpha_GMRES = opt_alpha
plt.plot(alphas,dp,'.-',lw=0.7)
plt.scatter(alpha_binarysearch,dp[alpha_idx],c='r',marker="*",zorder=5,label="Binary search")
plt.scatter(opt_alpha,DP(opt_alpha,g,sigma,theta,w,h,method,regulariser),c="g",marker="x",zorder=10,label="SciPy root")
ax = plt.gca()
ax.set_xscale('log')
plt.xlabel(r"$\alpha")
plt.ylabel("DP")
plt.legend()
format_plot('3c',show_axis=True)

## perona malik
regulariser = "anisotropic"
gamma=perona_malik(g.reshape(w,h))
f_alpha = unblur_gmres(g,alpha,sigma,w,h,regulariser,callback=counter(),gamma=gamma)
plot_deblur(f_alpha,ftrue,g,sigma,'4')

fi = g

for i in range(5):
    gamma=perona_malik(fi.reshape(w,h))
    fi = unblur_gmres(fi,alpha,sigma,w,h,regulariser,gamma=gamma)
    plot_deblur(fi,ftrue,g,sigma,'5i')




