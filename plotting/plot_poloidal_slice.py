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

nq, nr = data.shape

r = np.linspace(0.1, 14.5, nr)
q = np.linspace(0, 2*np.pi, nq, endpoint=False)

ax = plt.subplot(111, projection='polar')

data_min = data.min()
data_max = data.max()
if np.sign(data_min) != np.sign(data_max):
    bound = max(data_max, -data_min)
    clevels = np.linspace( -bound, bound, 101)
else:
    clevels = np.linspace( data_min, data_max, 101)
im = ax.contourf( q, r, data.T, clevels, cmap='seismic' )
#im = ax.contourf( q, r, data.T, clevels, cmap='jet' )
for c in im.collections:
    c.set_edgecolor('face')
plt.colorbar( im )
plt.title(f"t = {t}")

plt.show()
