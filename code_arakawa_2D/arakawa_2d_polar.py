import os
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye as sparse_id

import matplotlib.pyplot as plt

from utils_polar import plot_gridvals, plot_time_diag, make_movie
from discrete_brackets_polar import assemble_Jpp, assemble_Jpx, assemble_Jxp

#bracket = '++'
#bracket = '+x'
#bracket = 'x+'
bracket = 'akw'

explicit = False

N0_nodes0 = 40
N0_nodes1 = 50

Nt = 200

T = 1

f0_c=[0,-.5]
f0_s = .1

phi_c=[0,0]
phi_s = .5

nb_plots = 10
movie_duration = 2

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return [phi, rho]

def pol2cart(phi, rho):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return [x, y]

def init_f(rp):
    x = rp[1] * np.cos(rp[0])
    y = rp[1] * np.sin(rp[0])
    return np.exp(-((x-f0_c[0])**2+(y-f0_c[1])**2)/(2*f0_s**2))
    #return 0.1*np.exp(-((rp[1]-0.5)**2 + np.pi)/(2*f0_s**2))

def phi_ex(rp):
    x = rp[1] * np.cos(rp[0])
    y = rp[1] * np.sin(rp[0])
    return np.exp(-((x-phi_c[0])**2+(y-phi_c[1])**2)/(2*phi_s**2))
    #return np.exp(-((rp[1]-0.5)**2)/(2*f0_s**2))

# ---- ---- ---- ---- ---- ---- ---- ----
# grid
# indexing: f[i0 + i1*N0_nodes] = f(i0*h,i1*h)


grid0 = np.linspace(0, 2*np.pi, num=N0_nodes0) #phi
grid1 = np.linspace(0.01, 1.01, num=N0_nodes1) #r
print(grid0)
print(grid1)
dphi = (grid0[-1] - grid0[0])/(len(grid0)-1)
dr = (grid1[-1] - grid1[0])/(len(grid1)-1)

print(dr)
print(dphi)

N_nodes = N0_nodes0 * N0_nodes1

grid = np.array([[grid0[k%N0_nodes0], grid1[k//N0_nodes0]] for k in range(N_nodes)])
print(grid)
#grid_cart = np.array([pol2cart(grid[k][0], grid[k][1]) for k in range(len(grid))])
#grid0_cart = np.linspace(0, 1, num=N0_nodes) #x
#grid1_cart = np.linspace(-1, 1, num=N0_nodes) #y
#plt.scatter([grid_cart[k][0] for k in range(len(grid))], [grid_cart[k][1] for k in range(len(grid))] )
#plt.show()

dt = T/Nt

# ---- ---- ---- ---- ---- ---- ---- ----
# plots

method = bracket
if explicit:
    method += '_exp'
else:
    method += '_imp'

show_plots = False
plot_dir = './plots/polar_run_' + method + '/'
if not os.path.exists(plot_dir):
    print('creating directory '+plot_dir)
    os.makedirs(plot_dir)

if nb_plots > 0:
    plot_period = max(int(Nt/nb_plots),1)
else:
    plot_period = Nt

# movie_fn = plot_dir+'f_movie.gif'
movie_fn = plot_dir+'f_movie.mp4'
frames_list = []


# ---- ---- ---- ---- ---- ---- ---- ----
# operators
#scaling for the integrals
r_scaling = [grid1[k//N0_nodes0] for k in range(N_nodes)]
r_scaling_inv = [1/grid1[k//N0_nodes0] for k in range(N_nodes)]

#r_mat = np.diag(r_scaling)
#print(r_mat.shape)

def get_total_f(f):
   # f_s = f @ np.diag(r_scaling)
    return np.sum(f)

def get_total_f2(f):
   # f_s = f @ np.diag(r_scaling)
    f2 = np.multiply(f, f)   # pointwise multiplication
    return np.sum(f2)

def get_total_energy(f, phi):
   #f_s = np.multiply(r_scaling, f)
    phi_s = phi @ np.diag(r_scaling_inv)
    f_phi = np.multiply(f, phi_s)   # pointwise multiplication
    return np.sum(f_phi)


phi = np.array(list(map(phi_ex,grid)))


#X, Y = np.meshgrid(grid0, grid1)
#phi = phi.reshape(X.shape)
#fig = plt.figure()
#ax = fig.gca(polar=True)
#ax = plt.subplots(subplot_kw=dict(projection='polar'))
#ax.contour(X, Y, phi)
#fig.savefig('hi')
#plt.show()

plot_gridvals(grid0, grid1, [phi], f_labels=['phi'], title='phi', 
    show_plot=show_plots, plt_file_name=plot_dir+'phi.png')

#give the correct r-factor to the bracket
phi_hh = phi @ np.diag(r_scaling_inv) * 1/(4*dr*dphi)

# assemble discrete brackets as sparse matrices
# for f -> J(phi,f) = d_y phi * d_x f - d_x phi * d_y f

Jpp_phi = assemble_Jpp(phi_hh, N0_nodes0, N0_nodes1, grid1)
Jpx_phi = -assemble_Jpx(phi_hh, N0_nodes0, N0_nodes1, grid1)
Jxp_phi = -assemble_Jxp(phi_hh, N0_nodes0, N0_nodes1, grid1)

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
#print(J_phi.shape)
#print(r_scaling.shape)
print('tests:')
print(max(abs(J_phi @ np.ones(len(grid)))))
print(max(abs(J_phi @ np.diag(r_scaling_inv) @ phi)))

#J_phi = J_phi
#print(J_phi.shape)

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
    print(total_energy[nt])
    print(total_f[nt])
    print(total_f2[nt])
    print('tets')
    print(abs( sum( np.multiply(f, (J_phi.dot(f))) )))

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



