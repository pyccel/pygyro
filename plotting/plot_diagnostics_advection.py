import json
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')


def get_last_string(string: str):
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

    with open(foldername + method + "_adv_consv.txt") as file:
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

    assert k % 4 == 0

    entries = int(k / 4)
    for k in range(entries):
        assert data[2*k] == data[2 *
                                 (k + entries)], f'Labels not matching! {data[2*k]} != {data[2*(k + entries)]}'
        assert data[2*k +
                    1] == 'before', f'Was expecting label before but got {data[2*k + 1]}'
        assert data[2*(k + entries) +
                    1] == 'after', f'Was expecting label after but got {data[2*(k + entries) + 1]}'

    labels = []
    for k in range(entries):
        labels.append(data[2*k])

    print(f'labels are : ')
    for label in labels:
        print(f'\t{label}')

    data = np.array(data).reshape(-1, 2*entries)

    data = np.float64(data[2:])

    times = np.arange(0, data.shape[0]) * dt

    # Plot relative errors
    for k in range(entries):
        plt.plot(times, np.abs(np.divide(data[:, k] - data[:, k + entries], data[:, k])),
                 label=labels[k])
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('error')
    plt.title('Relative Errors for ' + method + ' Advection')

    if save_plot:
        plt.savefig(foldername + 'plots/' + method + '_adv_conservation.png')

    if show_plot:
        plt.show()

    plt.close()

    for k, label in enumerate(labels):
        if label == 'l2_phi':
            plt.plot(times, data[:, k], label=label)

            plt.yscale('log')
            plt.legend()
            plt.xlabel('time')
            plt.title('L2 norm of phi for ' + method + ' advection')

            if save_plot:
                plt.savefig(foldername + 'plots/' + method + '_adv_l2_phi.png')

            plt.close()

    mark = 0
    markers = {}
    for label in labels:
        if label[:3] == 'en_':
            substring = get_last_string(label)
            if substring is not None:
                if substring in markers.keys:
                    markers[substring][1] = mark
                else:
                    markers[substring] = [mark, None]
                mark += 1
            else:
                # if only one method of computing the energies was used
                markers = 'only one species'

    if len(markers) == 0:
        return

    elif markers == 'only one species':
        tot_en = np.zeros(np.shape(times))

    else:
        tot_en = {}
        for key in markers.keys:
            tot_en[key] = np.zeros(np.shape(times))

    # Plot energies
    for k, label in enumerate(labels):
        if label[:3] == 'en_':
            plt.plot(times, data[:, k], label=label)
            if markers == 'only one species':
                tot_en += data[:, k]
            else:
                key = get_last_string(label)
                tot_en[key] += data[:, k]

    if markers == 'only one species':
        plt.plot(times, tot_en, label='sum')
    else:
        for key in tot_en.keys:
            plt.plot(times, tot_en[key], label=key)

    plt.legend()
    plt.xlabel('time')
    plt.ylabel('energy')
    plt.title('Energies for ' + method + ' advection')

    if save_plot:
        plt.savefig(foldername + 'plots/' + method + '_adv_energies.png')

    if show_plot:
        plt.show()

    plt.close()


def main():
    """
    TODO
    """
    k = 2

    while True:
        foldername = 'simulation_' + str(k) + '/'
        if os.path.exists(foldername):
            if os.path.exists(foldername + 'akw_adv_consv.txt'):
                method = 'akw'
            elif os.path.exists(foldername + 'sl_adv_consv.txt'):
                method = 'sl'
            else:
                continue

            if not os.path.exists(foldername + 'plots/') and os.path.exists(foldername + method + '_adv_consv.txt'):
                os.mkdir(foldername + 'plots/')
                plot_diagnostics(foldername, method)
            k += 1
        else:
            break


if __name__ == '__main__':
    main()
