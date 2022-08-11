import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

filename = sys.argv[1]
f = h5py.File(filename, 'r')
data = np.array(f['dset'])
f.close()


t = os.path.splitext(os.path.basename(filename))[0].split('_')[1]

nq, nz = data.shape

z = np.linspace(0, 1506.7590666130668, nz, endpoint=False)
q = np.linspace(0, 2*np.pi, nq, endpoint=False)

ax = plt.subplot(111)

clevels = np.linspace( data.min(), data.max(), 101)
im = ax.contourf( q, z, data.T, clevels, cmap='jet' )
for c in im.collections:
    c.set_edgecolor('face')
plt.colorbar( im )
plt.title(f"t = {t}")
plt.xlabel('$\\theta$')
plt.ylabel('z')

plt.show()
