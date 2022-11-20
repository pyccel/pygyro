import os
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt


def plot_energies(foldername, save_plot=True, show_plot=False):
    """
    TODO
    """
    data = []

    with open(foldername + "phiDat.txt") as file:
        for line in file:
            for entry in line.split():
                data.append(entry)

    labels = ['time',
              'l2 norm of the electric potential',
              'l2 norm of the distribution function',
              'l1 norm of the distribution function',
              'number of particles',
              'minimum value of the distribution function',
              'maximum value of the distribution function',
              'Kinetic Energy',
              'Potential Energy']

    entries = len(labels)

    print(f'labels are : ')
    for label in labels:
        print(f'\t{label}')

    data = np.array(data, dtype=float).reshape(-1, entries)

    times = data[:, 0]

    tot_en = np.zeros(np.shape(times))

    # Plot energies
    for k, label in enumerate(labels):
        if label[-6:] == 'Energy':
            plt.plot(times, data[:, k], label=label)
            tot_en += data[:, k]

    plt.plot(times, tot_en, label='sum')

    plt.legend()
    plt.xlabel('time')
    plt.title('Energies')

    if save_plot:
        plt.savefig(foldername + 'plots/energies.png')

    if show_plot:
        plt.show()

    plt.close()


def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', metavar='foldername',
                        nargs='*', type=int)

    args = parser.parse_args()
    assert args.k is not None, "A number should be passed with -k to indicate for which folder it should be plotted for"
    k = args.k[0]

    foldername = 'simulation_' + str(k) + '/'
    if os.path.exists(foldername):
        if os.path.exists(foldername + 'phiDat.txt'):
            if not os.path.exists(foldername + 'plots/'):
                os.mkdir(foldername + 'plots/')
            print(f'Now plotting conservations for {foldername}')
            plot_energies(foldername)


if __name__ == '__main__':
    main()
