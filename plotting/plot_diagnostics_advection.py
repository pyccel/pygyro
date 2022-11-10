import os
import numpy as np
import matplotlib.pyplot as plt
import json


def plot_diagnostics(foldername, method, save_plot=True, show_plot=False):
    """
    Plot the relative differences of mass, l^2-norm, and energy against time from the
    'method'_consv.txt in foldername.

    Parameters:
        foldername : str
            name of the directory where the data is saved.

        method : str
            which method has been used for the advection step

        save_plot : bool
            if the plots should be saved

        show_plot : bool
            if the plots should be shown
    """
    data = []

    with open(foldername + "initParams.json") as file:
        dt = json.load(file)["dt"]

    with open(foldername + method + "_consv.txt") as file:
        for line in file:
            for entry in line.split():
                data.append(entry)

    data = np.array(data).reshape(-1, 8)

    data = np.float64(data[2:])

    times = np.arange(0, data.shape[0]) * dt

    # Plot relative errors
    plt.plot(times, np.abs(np.divide(data[:, 0] - data[:, 4], data[:, 0])),
             label='mass')
    plt.plot(times, np.abs(np.divide(data[:, 1] - data[:, 5], data[:, 1])),
             label='$l^2$-norm')
    plt.plot(times, np.abs(np.divide(data[:, 2] - data[:, 6], data[:, 2])),
             label='potential energy')
    plt.plot(times, np.abs(np.divide(data[:, 3] - data[:, 7], data[:, 3])),
             label='kinetic energy')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('error')
    plt.title('Relative error in mass, $l^2$-norm, and energy for ' + method + ' advection')

    if save_plot:
        plt.savefig(foldername + 'plots/' + method + '_conservation.png')

    if show_plot:
        plt.show()

    plt.close()

    # Plot energies
    plt.plot(times, data[:, 2], label='potential energy')
    plt.plot(times, data[:, 3], label='kinetic energy')
    plt.plot(times, data[:, 2] + data[:, 3], label='sum')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('energy')
    plt.title('Energies for ' + method + ' advection')

    if save_plot:
        plt.savefig(foldername + 'plots/' + method + '_energies.png')

    if show_plot:
        plt.show()

    plt.close()


def main():
    """
    TODO
    """
    k = 0

    while True:
        foldername = 'simulation_' + str(k) + '/'
        if os.path.exists(foldername):
            if os.path.exists(foldername + 'akw_consv.txt'):
                method = 'akw'
            elif os.path.exists(foldername + 'sl_consv.txt'):
                method = 'sl'
            else:
                continue

            if not os.path.exists(foldername + 'plots/') and os.path.exists(foldername + method + '_consv.txt'):
                os.mkdir(foldername + 'plots/')
                plot_diagnostics(foldername, method)
            k += 1
        else:
            break


if __name__ == '__main__':
    main()
