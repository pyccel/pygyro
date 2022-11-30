import numpy as np
import argparse
import matplotlib.pyplot as plt
import os


def plot_file(foldername):
    dataset = np.atleast_2d(np.loadtxt(os.path.join(foldername, 'phiDat.txt')))
            
    perm = np.array(dataset[:, 0]).argsort()

    shape = dataset.shape

    times = np.ndarray(shape[0])
    e_kin = np.ndarray(shape[0])
    e_pot = np.ndarray(shape[0])

    times[:] = dataset[perm, 0]
    e_kin[:] = dataset[perm, 7]
    e_pot[:] = dataset[perm, 8]
    e_tot = e_kin + e_pot

    plt.plot(times, e_kin, '-.', label="kinetic Energy")
    plt.plot(times, e_pot, '-.', label="potential Energy")
    plt.plot(times, e_tot, '-', label="total Energy")

parser = argparse.ArgumentParser(
    description='Plot the Energies as a function of time')

parser.add_argument('foldername', metavar='foldername', nargs='*', type=str,
                    help='The folders whose results should be plotted')


args = parser.parse_args()
filename = os.path.join(args.foldername[0], 'phiDat.txt')


plt.figure()

for f in args.foldername:
    plot_file(f)

plt.xlabel('time')
plt.grid()
plt.legend()
plt.savefig(args.foldername[0]+'energies.png')
plt.show()
