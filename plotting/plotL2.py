import numpy as np
import argparse
import matplotlib.pyplot as plt
import os


def plot_file(foldername, log_y : bool = True):
    dataset = np.atleast_2d(np.loadtxt(os.path.join(foldername, 'phiDat.txt')))

    shape = dataset.shape

    times = np.ndarray(shape[0])
    norm = np.ndarray(shape[0])

    times[:] = dataset[:, 0]
    norm[:] = dataset[:, 1]

    if log_y:
        plt.semilogy(times, norm, '.', label=foldername)
    else:
        plt.plot(times, norm, '.', label=foldername, color='orange')


def main():
    parser = argparse.ArgumentParser(
        description='Plot the L2 norm of phi as a function of time')
    parser.add_argument('-p', dest='pente', type=float,
                        default=3.54e-3,
                        help='The gradient of the expected linear regime (eg. for 1.12e-5*exp(3.83e-3*x) : 3.83e-3)')
    parser.add_argument('-m', dest='mult', type=float,
                        default=4e-5,
                        help='The multiplication factor of the expected linear regime (eg. for 1.12e-5*exp(3.83e-3*x) : 1.12e-5)')
    parser.add_argument('foldername', metavar='foldername', nargs='*', type=str,
                        help='The folders whose results should be plotted')

    args = parser.parse_args()
    filename = os.path.join(args.foldername[0], 'phiDat.txt')

    p = args.pente
    m = args.mult

    dataset = np.atleast_2d(np.loadtxt(filename))
    sorted_times = np.sort(dataset[:, 0])

    # plt.figure()
    plt.figure(figsize=(12, 5), dpi=250)
    ax1 = plt.subplot(1, 2, 1)
    ax1.semilogy(sorted_times, m*np.exp(p*sorted_times),
                label = str(m) + ' * exp('+str(p)+' * x)')
    for f in args.foldername:
        plot_file(f)
    plt.xlabel('time')
    plt.xlim([0, 4000])
    plt.ylim([0, 100])
    plt.ylabel('$\|\|\phi\|\|_2$')
    plt.grid()
    plt.legend()

    _ = plt.subplot(1, 2, 2)
    for f in args.foldername:
        plot_file(f, log_y=False)
    plt.xlim([4000, np.max(sorted_times)])
    plt.ylim([7, 10])
    plt.xlabel('time')
    plt.ylabel('$\|\|\phi\|\|_2$')
    plt.grid()
    plt.legend()
    
    plt.suptitle('$L^2$ - norm of $\phi$ with Arakawa scheme')
    plt.savefig(os.path.join(args.foldername[0], 'L2phi.png'))
    plt.close()


if __name__ == '__main__':
    main()
