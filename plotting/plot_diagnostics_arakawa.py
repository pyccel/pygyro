import os
import numpy as np
import matplotlib.pyplot as plt
import json


def plot_diagnostics(foldername, save_plot=True, show_plot=False):
    """
    Plot the relative differences of mass, l^2-norm, and energy against time from the
    akw_consv.txt in foldername.

    Parameters:
        foldername : str
            name of the directory where the data is saved.
        
        save_plot : bool
            if the plots should be saved
        
        show_plot : bool
            if the plots should be shown
    """
    data = []

    with open(foldername + "initParams.json") as file:
        dt = json.load(file)["dt"]

    with open(foldername + "akw_consv.txt") as file:
        for line in file:
            for entry in line.split():
                data.append(entry)

    data = np.array(data).reshape(-1, 6)

    data = np.float64(data[2:])

    times = np.arange(0, data.shape[0]) * dt

    plt.plot(times, np.abs(np.divide(data[:, 0] - data[:, 3], data[:, 0])),
             label='mass')
    plt.plot(times, np.abs(np.divide(data[:, 1] - data[:, 4], data[:, 1])),
             label='$l^2$-norm')
    plt.plot(times, np.abs(np.divide(data[:, 2] - data[:, 5], data[:, 2])),
             label='energy')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('error')
    plt.title('Relative error in mass, $l^2$-norm, and energy')

    if save_plot:
        plt.savefig(foldername + 'plots/akw_conservation.png')

    if show_plot:
        plt.show()


def main():
    """
    TODO
    """
    k = 0

    while True:
        foldername = 'simulation_' + str(k) + '/'
        if os.path.exists(foldername):
            if not os.path.exists(foldername + 'plots/') and os.path.exists(foldername + 'akw_consv.txt'):
                os.mkdir(foldername + 'plots/')
                plot_diagnostics(foldername)
            k += 1
        else:
            break


if __name__ == '__main__':
    main()
