import matplotlib.pyplot as plt
from utils.utils import imblur

def format_plot(name,show_axis=False):
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 144
    plt.rc('font', size=10)
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    plt.rc('xtick.major', size=4, width=1)
    plt.rc('ytick.major', size=4, width=1)
    plt.rc('axes', linewidth=1, labelsize='medium', titlesize='medium')
    if show_axis:
        plt.axhline(color='k', lw=0.5)
        plt.axvline(color='k', lw=0.5)
    plot_name = "figs/"+name
    plt.savefig(plot_name, bbox_inches='tight')
    plt.show();



def plot_deblur(f_unblurred,ftrue,g,sigma,plotname):
    w,h = ftrue.shape

    diff = f_unblurred - ftrue
    residual = g - imblur(f_unblurred,sigma,w,h)
    fig,ax = plt.subplots(1,3,figsize=(12,12))
    ax[0].imshow(g.reshape(w,h), cmap='gray')
    ax[0].set_title(f'Blurry image')
    ax[1].imshow(f_unblurred, cmap='gray')
    ax[1].set_title(f'Unblurred image')
    ax[2].imshow(residual.reshape(w,h), cmap='gray')
    ax[2].set_title('Residual $g-Af_{recon}$')
    

    format_plot(plotname);