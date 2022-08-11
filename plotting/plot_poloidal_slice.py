import h5py
import numpy as np
import matplotlib.pyplot    as plt
from matplotlib             import rc        as pltFont
import argparse
import os

from pygyro.tools.getSlice           import get_grid_slice
from pygyro.tools.getPhiSlice        import get_phi_slice
from pygyro.initialisation.constants import get_constants
from pygyro                          import splines as spl

parser = argparse.ArgumentParser(description='Process filename')
parser.add_argument('filename', metavar='filename',nargs=1,type=str,
                    default=[""],
                   help='the name of the folder from which to load and in which to save')

args = parser.parse_args()

filename = args.filename[0]
foldername = os.path.dirname(filename)

t = int(filename[filename.rfind('_')+1:filename.rfind('.')])
print("t=",t)

file = h5py.File(filename,'r')
dataset=file['/dset']

shape = dataset.shape
data_shape = list(shape)
data_shape[0]+=1

data = np.ndarray(data_shape)

data[:-1,:]=dataset[:]
data[-1,:]=data[0,:]

file.close()

constantFile = os.path.join(foldername, 'initParams.json')
if not os.path.exists(constantFile):
    raise RuntimeError("Can't find constants in simulation folder")

constants = get_constants(constantFile)

npts = constants.npts[:2]
degree = constants.splineDegrees[:2]
period = [False,True]
domain = [[constants.rMin, constants.rMax],[0,2*np.pi]]

nkts      = [n+1+d*(int(p)-1)              for (n,d,p)    in zip( npts,degree, period )]
breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
knots     = [spl.make_knots( b,d,p )       for (b,d,p)    in zip( breaks, degree, period )]
bsplines  = [spl.BSplines( k,d,p )         for (k,d,p)    in zip(  knots, degree, period )]
eta_grid  = [bspl.greville                 for bspl       in bsplines]

theta = np.repeat(np.append(eta_grid[1], 2*np.pi), npts[0]) \
                    .reshape(npts[1]+1, npts[0])
r = np.tile(eta_grid[0], npts[1]+1) \
        .reshape(npts[1]+1, npts[0])


x=r*np.cos(theta)
y=r*np.sin(theta)

font = {'size'   : 16}

pltFont('font', **font)
fig = plt.figure()
ax = fig.add_axes([0.1, 0.2, 0.7, 0.7],)
ax.set_title('T = {}'.format(t))
colorbarax2 = fig.add_axes([0.85, 0.15, 0.03, 0.8],)
plot = ax.pcolormesh(x,y,data,cmap="jet")
fig.colorbar(plot,cax=colorbarax2)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
#~ plt.savefig('debug_results/WO-Flux{}'.format(filename[filename.rfind('/'):-3]))
plt.show()
