
import numpy as np
import matplotlib.pyplot as plt

def plot_gridvals(grid0, grid1, fs, x0_label='x0', x1_label='x1', f_labels=None,
                  title=None, xticks=None, yticks=None,
                  show_plot=True, plt_file_name=None, show_colorbar=True,
                  vmin=None, vmax=None,
                  cmap='viridis'):
    """
    plot cartesian data fs = [f0, f1, ...]
    each density fn is an 1D array with values fn[i0 + i1*N0_nodes] = fn(i0,i1)
    with N0_nodes = len(grid0)
    :param grid0: 1d grid along x0
    :param grid1: 1d grid along x1
    :param fs: list of 1d arrays with density values
    """
    nf = len(fs)
    fig, ax = plt.subplots(1, nf, figsize=(10, 5), tight_layout=True, subplot_kw=dict(polar=True))
    if nf == 1:
        ax = [ax]
    N0_nodes = len(grid0)
    N1_nodes = len(grid1)
    if f_labels is None:
        f_labels = nf*[None]
    assert len(f_labels)==nf
    for n in range(nf):
        fn = fs[n].reshape(N1_nodes,N0_nodes)
        #ax[n] = ax[n].gca(polar = True)
        pcm = ax[n].pcolormesh(grid0, grid1, fn, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        #pcm = ax[n].pcolormesh(grid0, grid1, fn)
        if f_labels[n]:
            ax[n].set_title(f_labels[n], fontsize=10)
        if x0_label and x1_label:
            ax[n].set_xlabel(x0_label,fontsize=10)
            ax[n].set_ylabel(x1_label,fontsize=10)
        if xticks:
            ax[n].set_xticks(xticks)
        if yticks:
            ax[n].set_yticks(yticks)
        if show_colorbar:
            fig.colorbar(pcm, ax=ax[n])
    if title:
        plt.title(title)
    if plt_file_name:
        print("saving {} plot to file {}".format(title, plt_file_name))
        plt.savefig(plt_file_name)
    if show_plot:
        plt.show()
    else:
        plt.clf()
    plt.close() # or use ('all') ?



def plot_time_diag(diag, Nt, dt, plot_dir, name="diag"):
    fig2 = plt.figure()
    fig2.clf()
    fname = plot_dir+name+".png"
    message = "> plotting diag: " + name
    print(message)

    t_range = dt*np.array(range(Nt+1))
    plt.xlabel('t')

    # rho_min = min(rho_vals)
    # rho_max = max(rho_vals)
    # Y_min = u_min - 0.1*(u_max-u_min)
    # Y_max = u_max + 0.1*(u_max-u_min)
    # print(message) # + ", min/max = ", u_min, "/", u_max)
    # plt.xlim(0, 1)
    # plt.ylim(Y_min, Y_max)

    plt.title(name)
    # plt.legend(loc="upper center")
    plt.plot(t_range, diag, '-', color='k')
    fig2.savefig(fname)

    return fname

## movie
# see https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
import imageio.v2 as imageio

def make_movie(frames_list, movie_fn, frame_duration):
    print ("Creating movie {} from plots...".format(movie_fn))
    
    if movie_fn[-4:]=='.gif':
        writer = imageio.get_writer(movie_fn, mode='I', duration=frame_duration)
    elif movie_fn[-4:]=='.mp4':
        writer = imageio.get_writer(movie_fn, format='FFMPEG', mode='I', fps=1/frame_duration,
                    # codec='h264_vaapi',
                    # output_params=['-vaapi_device',
                    #                   '/dev/dri/renderD128',
                    #                   '-vf',
                    #                   'format=gray|nv12,hwupload'],
                    # pixelformat='vaapi_vld',
                    )
    else:
        raise NotImplementedError(movie_fn)

    for filename in frames_list:
        image = imageio.imread(filename)
        writer.append_data(image)

# w = iio.get_writer('my_video.mp4', format='FFMPEG', mode='I', fps=1,
#                        codec='h264_vaapi',
#                        output_params=['-vaapi_device',
#                                       '/dev/dri/renderD128',
#                                       '-vf',
#                                       'format=gray|nv12,hwupload'],
#                        pixelformat='vaapi_vld')


##
def get_total_f(dr, dtheta, r_scaling, f):
    """
    Compute the integral of f over the whole domain

    Parameters
    ----------
        f : array[float]
            array containing the values of f

    Returns
    -------
        float
            the sum over all values of f
    """
    f_s = np.multiply(r_scaling, f)

    return dr*dtheta*sum(f_s)


def get_total_f2(dr, dtheta, r_scaling, f):
    """
    Compute the integral of f^2 over the whole domain

    Parameters
    ----------
        f : array[float]
            array containing the values of f

    Returns
    -------
        float
            the sum over all squared values of f
    """
    f_s = np.multiply(r_scaling, f)

    # pointwise multiplication
    f2 = np.multiply(f, f_s)

    return dr*dtheta*sum(f2)


def get_total_energy(dr, dtheta, r_scaling, f, phi):
    """
    Compute the totoal energy, i.e. the integral of f times phi over the whole domain

    Parameters
    ----------
        f : array[float]
            array containing the values of f

        phi : array[float]
            array containing the values of phi

    Returns
    -------
        float
            the total energy
    """
    phi_s = np.multiply(r_scaling, phi)

    # pointwise multiplication
    f_phi = np.multiply(f, phi_s)

    return dr*dtheta*sum(f_phi)