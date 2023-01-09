import json
import os
import numpy as np
import argparse

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
        assert data[2*k] == data[2 * (k + entries)], \
            f'Labels not matching! {data[2*k]} != {data[2*(k + entries)]}'
        assert data[2*k + 1] == 'before', \
            f'Was expecting label before but got {data[2*k + 1]}'
        assert data[2*(k + entries) + 1] == 'after', \
            f'Was expecting label after but got {data[2*(k + entries) + 1]}'

    labels = []
    for k in range(entries):
        labels.append(data[2*k])

    print(f'labels are : ')
    for label in labels:
        print(f'\t{label}')

    data = np.array(data).reshape(-1, 2*entries)

    data = np.float64(data[2:])

    times = np.arange(0, data.shape[0]) * dt

    # ======================================
    # ======== Plot relative errors ========
    # ======================================
    for k in range(entries):
        plt.plot(times, np.abs(np.divide(data[:, k] - data[:, k + entries], data[:, k])),
                 label=labels[k])
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('error')
    plt.title('Relative Errors for ' + method + ' Advection')

    if save_plot:
        plt.savefig(foldername + 'plots/' + method + '_rel_err.png')

    if show_plot:
        plt.show()

    plt.close()

    # ======================================================
    # ======== Plot relative errors (without e_kin) ========
    # ======================================================
    for k in range(entries):
        if labels[k] != 'en_kin':
            plt.plot(times, np.abs(np.divide(data[:, k] - data[:, k + entries], data[:, k])),
                    label=labels[k])
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('error')
    plt.title('Relative Errors for ' + method + ' Advection')

    if save_plot:
        plt.savefig(foldername + 'plots/' + method + '_rel_err_wo_en_kin.png')

    if show_plot:
        plt.show()

    plt.close()

    # ======================================
    # ======== Plot absolute errors ========
    # ======================================
    for k in range(entries):
        plt.plot(times, data[:, k] - data[:, k + entries],
                 label=labels[k])
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('error')
    plt.title('Absolute Errors for ' + method + ' Advection')

    if save_plot:
        plt.savefig(foldername + 'plots/' + method + '_abs_err.png')

    if show_plot:
        plt.show()

    plt.close()

    # =====================================
    # ======== Plot l2-norm of phi ========
    # =====================================
    for k, label in enumerate(labels):
        if label == 'l2_phi':
            plt.plot(times, data[:, k], label=label)

            plt.yscale('log')
            plt.legend()
            plt.xlabel('time')
            plt.title('L2 norm of phi for ' + method + ' advection')

            if save_plot:
                plt.savefig(foldername + 'plots/' + method + '_l2_phi.png')

            plt.close()

    tot_en = np.zeros(np.shape(times))

    # ===============================
    # ======== Plot energies ========
    # ===============================
    for k, label in enumerate(labels):
        if label[:3] == 'en_':
            plt.plot(times, data[:, k], label=label)
            tot_en += data[:, k]

    plt.plot(times, tot_en, label='sum')

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', metavar='foldername',
                        nargs='*', type=int)
    parser.add_argument('-cluster', metavar='foldername',
                        nargs='*', type=str)

    args = parser.parse_args()
    if args.k is not None:
        k = args.k[0]
    else:
        k = 0

    if args.cluster is not None:
        cluster = args.cluster[0]
    else:
        cluster = None

    while True:
        if cluster is not None:
            foldername = cluster + '/sim_' + str(k) + '/'
        else:
            foldername = 'simulation_' + str(k) + '/'
        if os.path.exists(foldername):
            if os.path.exists(foldername + 'akw_adv_consv.txt'):
                method = 'akw'
            elif os.path.exists(foldername + 'sl_adv_consv.txt'):
                method = 'sl'
            else:
                continue

            if os.path.exists(foldername + method + '_adv_consv.txt'):
                if not os.path.exists(foldername + 'plots/'):
                    os.mkdir(foldername + 'plots/')
                plot_diagnostics(foldername, method, show_plot=True)
            # k += 1
            break
        else:
            break


if __name__ == '__main__':
    main()
