import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import argparse


def get_last_string(string : str):
    """
    Get the last piece of a string after an underscore
    """
    assert isinstance(string, str), 'Object is not a string!'

    underscores = string.count('_')

    if underscores == 0:
        return string
    elif underscores == 1:
        return None
    else:
        last_index = string.rindex('_')
        return string[last_index:]


def plot_diagnostics(foldername, method, save_plot=True, show_plot=False):
    """
    Plot the relative errors of quantities in the 'method'_consv.txt
    in foldername against time.

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

    k = 0
    while True:
        try:
            np.float64(data[k])
        except:
            k += 1
        else:
            break

    entries = int(k)

    labels = []
    for k in range(1, entries):
        labels.append(data[k])

    print(f'labels are : ')
    for label in labels:
        print(f'\t{label}')

    data = np.array(data).reshape(-1, entries)

    data = np.float64(data[2:])

    times = data[:, 0]

    data = data[:, 1:entries]

    a = 4e-5
    b = 0.00354

    for k, label in enumerate(labels):
        if label == 'l2_phi':
            dataset = np.atleast_2d(np.loadtxt(os.path.join(foldername, 'phiDat.txt')))

            shape = dataset.shape

            times2 = np.ndarray(shape[0])
            norm = np.ndarray(shape[0])

            times2[:] = dataset[:, 0]
            norm[:] = dataset[:, 1]

            plt.plot(times, data[:, k], label=label)
            plt.plot(times2, norm, 'co', label='saved')
            plt.plot(times, a * np.exp(b * times), label='analytical')

            plt.legend()
            plt.xlabel('time')
            plt.yscale('log')
            plt.title('L2 norm of phi for ' + method + ' advection, data saved during the simulation, and analytical linear growth rate')

            if save_plot:
                plt.savefig(foldername + 'plots/' + method + '_l2_phi.png')

            plt.close()

    tot_en = np.zeros(np.shape(times))

    # Plot energies
    for k, label in enumerate(labels):
        if label[:3] == 'en_':
            plt.plot(times, data[:, k], label=label)
            tot_en += data[:, k]
            # plt.plot(times, data[:, k] / np.max(np.abs(data[:, k])), label=label)
            # tot_en += data[:, k] / np.max(np.abs(data[:, k]))

    plt.plot(times, tot_en, label='sum')

    plt.legend()
    plt.xlabel('time')
    plt.ylabel('energy')
    # plt.ylim([-50, 50])
    plt.title('Energies for ' + method + ' advection')

    if save_plot:
        plt.savefig(foldername + 'plots/' + method + '_energies.png')

    if show_plot:
        plt.show()

    plt.close()

    # Plot rest
    for k, label in enumerate(labels):
        if label[:3] != 'en_' and label != 'l2_phi':
            plt.plot(times, data[:, k], label=label)

    plt.legend()
    plt.xlabel('time')
    plt.yscale('log')
    plt.title('Quantities for ' + method + ' advection')

    if save_plot:
        plt.savefig(foldername + 'plots/' + method + '_quantities.png')

    if show_plot:
        plt.show()

    plt.close()

    # Plot relative error for rest
    for k, label in enumerate(labels):
        if label[:3] != 'en_' and label != 'l2_phi':
            plt.plot(times, np.abs(np.divide(data[:, k] - data[0, k], data[0, k])), label=label)

    plt.legend()
    plt.xlabel('time')
    plt.title('Relative Errors for Quantities for ' + method + ' advection')

    if save_plot:
        plt.savefig(foldername + 'plots/' + method + '_quantities_rel_err.png')

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
    k = args.k[0]

    while True:
        foldername = 'simulation_' + str(k) + '/'
        if os.path.exists(foldername):
            if os.path.exists(foldername + 'akw_consv.txt'):
                method = 'akw'
            elif os.path.exists(foldername + 'sl_consv.txt'):
                method = 'sl'
            else:
                continue

            if os.path.exists(foldername + method + '_consv.txt'):
                if not os.path.exists(foldername + 'plots/'):
                    os.mkdir(foldername + 'plots/')
                print(f'Now plotting conservations for {foldername}')
                plot_diagnostics(foldername, method)
            # k += 1
            break
        else:
            break


if __name__ == '__main__':
    main()
