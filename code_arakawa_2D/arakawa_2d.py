import os
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye as sparse_id

from utils import plot_gridvals, plot_time_diag, make_movie
from discrete_brackets import assemble_Jpp, assemble_Jpx, assemble_Jxp

# bracket = '++'
# bracket = '+x'
# bracket = 'x+'
bracket = 'akw'

explicit = False

N0_nodes = 50
Nt = 100

T = .5

f0_c=[.2,.5]
f0_s = .07

phi_c=[.5,.5]
phi_s = .5

nb_plots = 10
movie_duration = 2

def init_f(x):
    return np.exp(-((x[0]-f0_c[0])**2+(x[1]-f0_c[1])**2)/(2*f0_s**2))

def phi_ex(x):
    return np.exp(-((x[0]-phi_c[0])**2+(x[1]-phi_c[1])**2)/(2*phi_s**2))


# ---- ---- ---- ---- ---- ---- ---- ----
# grid
# indexing: f[i0 + i1*N0_nodes] = f(i0*h,i1*h)

h_grid = 1/(N0_nodes-1)
grid0 = np.linspace(0, 1, num=N0_nodes)
grid1 = np.linspace(0, 1, num=N0_nodes)
N_nodes = N0_nodes * N0_nodes
grid = np.array([[grid0[k%N0_nodes], grid1[k//N0_nodes]] for k in range(N_nodes)])

dt = T/Nt

# ---- ---- ---- ---- ---- ---- ---- ----
# plots

method = bracket
if explicit:
    method += '_exp'
else:
    method += '_imp'

show_plots = False
plot_dir = './plots/cart_run_' + method + '/'
if not os.path.exists(plot_dir):
    print('creating directory '+plot_dir)
    os.makedirs(plot_dir)

if nb_plots > 0:
    plot_period = max(int(Nt/nb_plots),1)
else:
    plot_period = Nt

# movie_fn = plot_dir+'f_movie.gif'
movie_fn = plot_dir + 'f_movie.mp4'
frames_list = []


# ---- ---- ---- ---- ---- ---- ---- ----
# operators

def get_total_f(f):
    return np.sum(f)

def get_total_f2(f):
    f2 = np.multiply(f, f)   # pointwise multiplication
    return np.sum(f2)

def get_total_energy(f, phi):
    f_phi = np.multiply(f, phi)   # pointwise multiplication
    return np.sum(f_phi)



# phi on grid
phi = np.array(list(map(phi_ex,grid)))
plot_gridvals(grid0, grid1, [phi], f_labels=['phi'], title='phi', 
    show_plot=show_plots, plt_file_name=plot_dir+'phi.png')

phi_hh = phi/(2*h_grid)**2

# assemble discrete brackets as sparse matrices
# for f -> J(phi,f) = d_y phi * d_x f - d_x phi * d_y f

Jpp_phi = assemble_Jpp(phi_hh, N0_nodes)
Jpx_phi = -assemble_Jpx(phi_hh, N0_nodes)
Jxp_phi = -assemble_Jxp(phi_hh, N0_nodes)

if bracket == '++':
    J_phi = Jpp_phi
elif bracket == '+x':
    J_phi = Jpx_phi
elif bracket == 'x+':
    J_phi = Jxp_phi
elif bracket == 'akw':
    J_phi = (Jpp_phi + Jpx_phi + Jxp_phi)/3
else:
    raise NotImplementedError(bracket)

if explicit:
    I = A = B = None
else:
    I = sparse_id( N_nodes )
    A = I - dt/2*J_phi 
    B = I + dt/2*J_phi 

# f0 on grid
f = np.array(list(map(init_f,grid)))
plt_fn = plot_dir+'f0.png'
plot_gridvals(grid0, grid1, [f], f_labels=['f0'], title='f0', 
    show_plot=show_plots, plt_file_name=plt_fn)
frames_list.append(plt_fn)

# diags
total_f = np.zeros(Nt+1)
total_energy = np.zeros(Nt+1)
total_f2 = np.zeros(Nt+1)

total_energy[0] = get_total_energy(f,phi)
total_f[0] = get_total_f(f)
total_f2[0] = get_total_f2(f)

for nt in range(Nt):
    print("computing f^n for n = {}".format(nt+1))
    
    total_energy[nt+1] = get_total_energy(f,phi)
    total_f[nt+1] = get_total_f(f)
    total_f2[nt+1] = get_total_f2(f)

    if explicit:
        f[:] += dt*J_phi.dot(f)
    else:
        # print('solving source problem with scipy.spsolve...')
        f[:] = spsolve(A, B.dot(f))

    # visualisation
    nt_vis = nt+1
    if nt_vis%plot_period == 0 or nt_vis == Nt:
        plt_fn = plot_dir+'f_{}.png'.format(nt_vis)
        plot_gridvals(grid0, grid1, [f], f_labels=['f_end'], title='f end', 
            show_plot=show_plots, plt_file_name=plt_fn)
        frames_list.append(plt_fn)


plot_time_diag(total_energy, Nt, dt, plot_dir, name="energy")
plot_time_diag(total_f, Nt, dt, plot_dir, name="f")
plot_time_diag(total_f2, Nt, dt, plot_dir, name="f2")

make_movie(frames_list, movie_fn, frame_duration=movie_duration/nb_plots)



