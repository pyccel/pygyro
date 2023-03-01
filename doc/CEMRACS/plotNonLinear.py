import numpy as np
import argparse
import matplotlib.pyplot as plt
import os


def plot_file(foldername, label):
    dataset = np.atleast_2d(np.loadtxt(os.path.join(foldername, 'phiDat.txt')))

    shape = dataset.shape

    times = np.ndarray(shape[0])
    norm = np.ndarray(shape[0])

    times[:] = dataset[:, 0]
    norm[:] = dataset[:, 1]

    idx =np.argsort(times)

    times = times[idx][2000:]
    norm = norm[idx][2000:]

    plt.plot(times, norm, '-', label=label)


#dataset = np.atleast_2d(np.loadtxt(filename))
#sorted_times = np.sort(dataset[:, 0])

fig, ax = plt.subplots(1,1)
ax.plot([],[])
plot_file('iota0', '$\iota=0.0$')
plot_file('iota8', '$\iota=0.8$')
plt.xlabel('t')
plt.ylabel(r'$|\phi|_2$')
plt.grid()
plt.legend()
fig.tight_layout()
fig.savefig('non_linear_phase.png')
plt.show()
