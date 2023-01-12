import json
import os
import numpy as np
import argparse

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_labels(raw_label):
    if raw_label == 'mass_f':
        return r'Mass of $f$'
    elif raw_label == 'l2_f':
        return r'$L^2$-norm of $f$'
    elif raw_label == 'l2_phi':
        return raw_label
    elif raw_label == 'en_kin':
        return 'Kinetic Energy'
    elif raw_label == 'en_pot':
        return 'Potential Energy'
    else:
        raise ValueError(f'Got {raw_label} as raw_label which is not a valid raw label!')


def plot_diagnostics(foldername, folders, methods, plot_labels, styles=None, save_plot=True, show_plot=False):
    """
    Plot the relative errors of quantities in the 'method'_consv.txt
    in foldername against time.

    Parameters:
    -----------
    foldername : str
        name of the directory where the data is saved.

    folders : list
        list of folders where the data points that should be compared are

    save_plot : bool
        if the plots should be saved

    show_plot : bool
        if the plots should be shown
    """
    assert len(methods) == len(folders)

    if styles is None:
        styles = ['solid'] * len(folders)
    else:
        assert len(styles) == len(folders)

    data = []
    dt = []
    labels = []
    times = []

    for (folder, method) in zip(folders, methods):
        data.append([])
        with open(folder + "initParams.json") as file:
            dt += [json.load(file)["dt"]]

        with open(folder + method + "_adv_consv.txt") as file:
            for line in file:
                for entry in line.split():
                    data[-1].append(entry)

    # Get labels
    entries = [0] * len(methods)
    for i in range(len(methods)):
        k = 0
        while True:
            try:
                np.float64(data[i][k])
            except:
                k += 1
            else:
                break

        assert k % 4 == 0

        entries[i] = int(k / 4)

        for k in range(entries[i]):
            assert data[i][2*k] == data[i][2*(k + entries[i])], \
                f'Labels not matching! {data[i][2*k]} != {data[i][2*(k + entries[i])]}'
            assert data[i][2*k + 1] == 'before', \
                f'Was expecting label before but got {data[i][2*k + 1]}'
            assert data[i][2*(k + entries[i]) + 1] == 'after', \
                f'Was expecting label after but got {data[i][2*(k + entries[i]) + 1]}'

        labels.append([])
        for k in range(entries[i]):
            labels[-1].append(get_labels(data[i][2*k]))

        data[i] = np.array(data[i]).reshape(-1, 2*entries[i])

        data[i] = np.float64(data[i][2:])

        times += [np.arange(0, data[i].shape[0]) * dt[i]]

    for i in range(len(labels)):
        assert labels[0][i] == labels[1][i]

    print('The labels are : ')
    for label in labels[0]:
        print(f'   {label}')

    # ======================================
    # ======== Plot relative errors ========
    # ======================================
    plt.figure(figsize=(9.5, 7), dpi=250)
    p = 1
    for i in range(entries[0]):
        if labels[0][i] == 'l2_phi':
            continue
        else:
            plt.subplot(2, 2, p)
            p += 1
            for k in range(len(methods)):
                plt.plot(times[k], np.abs(np.divide(data[k][:, i] - data[k][:, i + entries[k]], data[k][:, i])),
                         label=plot_labels[k], linestyle=styles[k])
            plt.legend()
        plt.subplots_adjust(left=0.05, right=0.99)
        plt.title(labels[k][i])
        if labels[k][i] == 'Kinetic Energy':
            plt.yscale('log')

    if dt[0] == 2:
        plt.suptitle(r'Relative Errors for Poloidal Advection Step ($\Delta t = 2$)')
    else:
        plt.suptitle(r'Relative Errors for Poloidal Advection Step')

    if save_plot:
        plt.savefig(foldername + 'rel_err.png')

    if show_plot:
        plt.show()

    plt.close()

    # ==================================================
    # ======== Plot relative errors (log scale) ========
    # ==================================================
    plt.figure(figsize=(9.5, 7), dpi=250)
    p = 1
    for i in range(entries[0]):
        if labels[0][i] == 'l2_phi':
            continue
        else:
            plt.subplot(2, 2, p)
            p += 1
            for k in range(len(methods)):
                plt.plot(times[k], np.abs(np.divide(data[k][:, i] - data[k][:, i + entries[k]], data[k][:, i])),
                         label=plot_labels[k], linestyle=styles[k])
            plt.legend()
        plt.subplots_adjust(left=0.05, right=0.99)
        plt.title(labels[k][i])
        plt.yscale('log')
    if dt[0] == 2:
        plt.suptitle(r'Relative Errors for Poloidal Advection Step ($\Delta t = 2$)')
    else:
        plt.suptitle(r'Relative Errors for Poloidal Advection Step')

    if save_plot:
        plt.savefig(foldername + 'rel_err_log.png')

    if show_plot:
        plt.show()

    plt.close()

    # ======================================
    # ======== Plot absolute errors ========
    # ======================================
    plt.figure(figsize=(9.5, 7), dpi=250)
    p = 1
    for i in range(entries[0]):
        if labels[0][i] == 'l2_phi':
            continue
        else:
            plt.subplot(2, 2, p)
            p += 1
            for k in range(len(methods)):
                plt.plot(times[k], np.abs(data[k][:, i] - data[k][:, i + entries[k]]),
                         label=plot_labels[k], linestyle=styles[k])
            plt.legend()
        plt.subplots_adjust(left=0.08, right=0.98)
        plt.title(labels[k][i])
    if dt[0] == 2:
        plt.suptitle(r'Absolute Errors for Poloidal Advection Step ($\Delta t = 2$)')
    else:
        plt.suptitle(r'Absolute Errors for Poloidal Advection Step')

    if save_plot:
        plt.savefig(foldername + 'abs_err.png')

    if show_plot:
        plt.show()

    plt.close()

    # ==================================================
    # ======== Plot absolute errors (log scale) ========
    # ==================================================
    plt.figure(figsize=(9.5, 7), dpi=250)
    p = 1
    for i in range(entries[0]):
        if labels[0][i] == 'l2_phi':
            continue
        else:
            plt.subplot(2, 2, p)
            p += 1
            for k in range(len(methods)):
                plt.plot(times[k], np.abs(data[k][:, i] - data[k][:, i + entries[k]]),
                         label=plot_labels[k], linestyle=styles[k])
            plt.legend()
        plt.subplots_adjust(left=0.08, right=0.98)
        plt.title(labels[k][i])
        plt.yscale('log')
    if dt[0] == 2:
        plt.suptitle(r'Absolute Errors for Poloidal Advection Step ($\Delta t = 2$)')
    else:
        plt.suptitle(r'Absolute Errors for Poloidal Advection Step')

    if save_plot:
        plt.savefig(foldername + 'abs_err_log.png')

    if show_plot:
        plt.show()

    plt.close()


def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-cobra', metavar='foldername',
                        nargs='*', type=int)
    parser.add_argument('-raven', metavar='foldername',
                        nargs='*', type=int)

    args = parser.parse_args()
    if args.cobra is not None:
        cobra = args.cobra[0]
    else:
        cobra = 0

    if args.raven is not None:
        raven = args.raven[0]
    else:
        raven = 0

    k = 0

    parentpath = 'plotting/comparison/'
    if not os.path.exists(parentpath):
        os.mkdir(parentpath)

    while True:
        folder_cobra = 'cobra/sim_' + str(cobra) + '/'
        # folder_cobra = 'raven/sim_' + str(cobra) + '/'
        folder_raven = 'raven/sim_' + str(raven) + '/'
        foldername = parentpath + str(k) + '/'
        if os.path.exists(foldername):
            k += 1
        else:
            os.mkdir(foldername)
            plot_diagnostics(foldername,
                             [folder_cobra, folder_raven],
                             ['sl', 'akw'],
                            #  ['akw', 'akw'],
                             plot_labels = ['Semi-Lagrangian', 'Arakawa'],
                            #  plot_labels = [r'$\Delta t = 1$', r'$\Delta t = 2$'],
                             styles=['solid', 'dashdot'])
            break


if __name__ == '__main__':
    main()
