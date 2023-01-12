import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import argparse

from plot_versus import get_labels


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


def plot_diagnostics_old(foldername, method, save_plot=True, show_plot=False):
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

            plt.plot(times2, norm, 'co', label='saved')
            plt.plot(times, data[:, k], label=label)
            plt.plot(times, a * np.exp(b * times), label='analytical')

            plt.legend()
            plt.xlabel('time')
            plt.yscale('log')
            plt.title('L2 norm of phi for ' + method + ' advection, data saved during the simulation,\n and analytical linear growth rate')

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


def plot_diagnostics_new(foldername, method, save_plot=True, show_plot=False):
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
    labels = []
    plot_labels = []

    with open(foldername + "initParams.json") as file:
        dt = json.load(file)["dt"]

    with open(foldername + method + "_adv_consv.txt") as file:
        for line in file:
            for entry in line.split():
                data.append(entry)

    # Get labels
    k = 0
    j = 0
    while True:
        if data[k] == 'z0v0':
            k += 1
            j += 1
        try:
            np.float64(data[k])
        except:
            k += 1
        else:
            break

    k -= j
    assert k % 8 == 0

    entries = int(k / 8)

    for k in range(entries):
        assert data[2*k] == data[2*(k + 2*entries)], \
            f'Labels not matching! {data[2*k]} != {data[2*(k + 2*entries)]}'
        assert data[2*k + 1] == 'before', \
            f'Was expecting label before but got {data[2*k + 1]}'
        assert data[2*(k + 2*entries) + 1] == 'after', \
            f'Was expecting label after but got {data[2*(k + 2*entries) + 1]}'

    for k in range(entries):
        labels.append(data[2*k])
        plot_labels.append(get_labels(data[2*k]))

    data = np.array(data).reshape(-1, 4*entries)

    data = np.float64(data[2:])


    print('The plot labels are : ')
    for label in plot_labels:
        print(f'   {label}')

    data = np.array(data).reshape(-1, 4*entries)

    data = np.float64(data[1:])

    times = np.arange(0, data.shape[0]) * dt

    tot_en = np.zeros(np.shape(times))

    # Plot energies
    for k, label in enumerate(labels):
        if label[:3] == 'en_':
            plt.plot(times, data[:, k], label=plot_labels[k])
            tot_en += data[:, k]

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
            plt.plot(times, data[:, k], label=plot_labels[k])

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
    parser.add_argument('--old', action='store_true')

    args = parser.parse_args()
    k = args.k[0]
    old = args.old

    while True:
        foldername = 'simulation_' + str(k) + '/'
        if os.path.exists(foldername):
            if os.path.exists(foldername + 'akw_adv_consv.txt'):
                method = 'akw'
            elif os.path.exists(foldername + 'sl_adv_consv.txt'):
                method = 'sl'
            else:
                k += 1
                continue

            if os.path.exists(foldername + method + '_adv_consv.txt'):
                if not os.path.exists(foldername + 'plots/'):
                    os.mkdir(foldername + 'plots/')
                print(f'Now plotting conservations for {foldername}')
                if old:
                    plot_diagnostics_old(foldername, method)
                else:
                    plot_diagnostics_new(foldername, method)

            # k += 1
            break
        else:
            break


if __name__ == '__main__':
    main()