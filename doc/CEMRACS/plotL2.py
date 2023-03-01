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

    times = times[idx][:2001]
    norm = norm[idx][:2001]

    plt.semilogy(times, norm, '-', label=label)


parser = argparse.ArgumentParser(
    description='Plot the L2 norm of phi as a function of time')
parser.add_argument('-p', dest='pente', type=float,
                    default=3.83e-3,
                    help='The gradient of the expected linear regime (eg. for 1.12e-5*exp(3.83e-3*x) : 3.83e-3)')
parser.add_argument('-m', dest='mult', type=float,
                    default=1.12e-5,
                    help='The multiplication factor of the expected linear regime (eg. for 1.12e-5*exp(3.83e-3*x) : 1.12e-5)')
parser.add_argument('foldername', metavar='foldername', nargs='*', type=str,
                    help='The folders whose results should be plotted')

args = parser.parse_args()
#filename = os.path.join(args.foldername[0], 'phiDat.txt')

p = args.pente
m = args.mult

#dataset = np.atleast_2d(np.loadtxt(filename))
#sorted_times = np.sort(dataset[:, 0])
sorted_times = np.array([0, 4000])

fig, ax = plt.subplots(1,1)
ax.semilogy(sorted_times, m*np.exp(p*sorted_times),
             label=r'$1.12\cdot 10^{-5} \exp(3.83\cdot 10^{-3} t)$')
plot_file('iota0', '$\iota=0.0$')
plot_file('iota8', '$\iota=0.8$')
plt.xlabel('t')
plt.ylabel(r'$|\phi|_2$')
plt.grid()
plt.legend()
fig.tight_layout()
fig.savefig('growth_rate.png')
plt.show()
