import os
import numpy as np
import matplotlib.pyplot as plt
import json


def plot_diagnostics(foldername, v_loc, save_plot=True, show_plot=False):
    """
    Plot the relative differences of mass, l^2-norm, and energy against time from the
    akw_consv.txt in foldername.

    Parameters:
        foldername : str
            name of the directory where the data is saved.

        v_loc : str
            '0' or 'max'; where in the velocity distribution the slice (idx_v) is

        save_plot : bool
            if the plots should be saved

        show_plot : bool
            if the plots should be shown
    """
    data = []

    with open(foldername + "initParams.json") as file:
        dt = json.load(file)["dt"]

    if v_loc == '0':
        with open(foldername + "akw_consv_v_0.txt") as file:
            for line in file:
                for entry in line.split():
                    data.append(entry)
    elif v_loc == 'max':
        with open(foldername + "akw_consv_v_max.txt") as file:
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
             label='potential energy')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('error')
    plt.title('Relative error in mass, $l^2$-norm, and potential energy')

    if save_plot:
        if v_loc == '0':
            plt.savefig(foldername + 'plots_v_0/akw_consv_rel_err.png')
        elif v_loc == 'max':
            plt.savefig(foldername + 'plots_v_max/akw_consv_rel_err.png')

    if show_plot:
        plt.show()

    plt.close()

    plt.plot(times, data[:, 2], label='potential energy')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel(r'E_{pot}')
    plt.title('The Potential Energy as a Function of time')

    if save_plot:
        if v_loc == '0':
            plt.savefig(foldername + 'plots_v_0/akw_consv_en_pot.png')
        elif v_loc == 'max':
            plt.savefig(foldername + 'plots_v_max/akw_consv_en_pot.png')

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
            if not os.path.exists(foldername + 'plots_v_0/') and os.path.exists(foldername + 'akw_consv_v_0.txt'):
                os.mkdir(foldername + 'plots_v_0/')
                plot_diagnostics(foldername, '0')
            if not os.path.exists(foldername + 'plots_v_max/') and os.path.exists(foldername + 'akw_consv_v_max.txt'):
                os.mkdir(foldername + 'plots_v_max/')
                plot_diagnostics(foldername, 'max')
            k += 1
        else:
            break


if __name__ == '__main__':
    main()
