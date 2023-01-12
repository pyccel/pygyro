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
        raise ValueError(
            f'Got {raw_label} as raw_label which is not a valid raw label!')


def plot_diagnostics_old(foldername, folders, methods, plot_labels, styles=None, save_plot=True, show_plot=False):
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
        plt.suptitle(
            r'Relative Errors for Poloidal Advection Step ($\Delta t = 2$)')
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
        plt.suptitle(
            r'Relative Errors for Poloidal Advection Step ($\Delta t = 2$)')
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
        plt.suptitle(
            r'Absolute Errors for Poloidal Advection Step ($\Delta t = 2$)')
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
        plt.suptitle(
            r'Absolute Errors for Poloidal Advection Step ($\Delta t = 2$)')
    else:
        plt.suptitle(r'Absolute Errors for Poloidal Advection Step')

    if save_plot:
        plt.savefig(foldername + 'abs_err_log.png')

    if show_plot:
        plt.show()

    plt.close()


def plot_diagnostics_new(foldername, folders, methods, plot_labels, styles=None, save_plot=True, show_plot=False):
    """
    Plot the relative errors of quantities in the 'method'_consv.txt
    in foldername against time. New format where conservation file
    also contains (z,v) = (0,0) slice conservation properties.

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
        j = 0
        while True:
            if data[i][k] == 'z0v0':
                k += 1
                j += 1
            try:
                np.float64(data[i][k])
            except:
                k += 1
            else:
                break

        k -= j
        assert k % 8 == 0

        entries[i] = int(k / 8)

        for k in range(entries[i]):
            assert data[i][2*k] == data[i][2*(k + 2*entries[i])], \
                f'Labels not matching! {data[i][2*k]} != {data[i][2*(k + 2*entries[i])]}'
            assert data[i][2*k + 1] == 'before', \
                f'Was expecting label before but got {data[i][2*k + 1]}'
            assert data[i][2*(k + 2*entries[i]) + 1] == 'after', \
                f'Was expecting label after but got {data[i][2*(k + 2*entries[i]) + 1]}'

        labels.append([])
        for k in range(entries[i]):
            labels[-1].append(get_labels(data[i][2*k]))

        data[i] = np.array(data[i]).reshape(-1, 4*entries[i])

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
    # totally integrated
    plt.figure(figsize=(9.5, 7), dpi=250)
    p = 1
    for i in range(entries[0]):
        if labels[0][i] == 'l2_phi':
            continue
        else:
            plt.subplot(2, 2, p)
            p += 1
            for k in range(len(methods)):
                plt.plot(times[k], np.abs(np.divide(data[k][:, i] - data[k][:, i + 2*entries[k]], data[k][:, i])),
                         label=plot_labels[k], linestyle=styles[k])
            plt.legend()
        plt.subplots_adjust(left=0.05, right=0.99)
        plt.title(labels[k][i])
        if labels[k][i] == 'Kinetic Energy':
            plt.yscale('log')

    if dt[0] == 2:
        plt.suptitle(
            r'Relative Errors for Poloidal Advection Step ($\Delta t = 2$)')
    else:
        plt.suptitle(r'Relative Errors for Poloidal Advection Step')

    if save_plot:
        plt.savefig(foldername + 'rel_err.png')

    if show_plot:
        plt.show()

    plt.close()

    # (z,v) = (0,0) slice
    plt.figure(figsize=(9.5, 7), dpi=250)
    p = 1
    for i in range(entries[0]):
        if labels[0][i] == 'l2_phi':
            continue
        else:
            plt.subplot(2, 2, p)
            p += 1
            for k in range(len(methods)):
                plt.plot(times[k], np.abs(np.divide(data[k][:, i + entries[k]] - data[k][:, i + 3*entries[k]], data[k][:, i + entries[k]])),
                         label=plot_labels[k], linestyle=styles[k])
            plt.legend()
        plt.subplots_adjust(left=0.05, right=0.99)
        plt.title(labels[k][i])
        if labels[k][i] == 'Kinetic Energy':
            plt.yscale('log')

    if dt[0] == 2:
        plt.suptitle(
            r'Relative Errors for Poloidal Advection Step on $(z,v) = (0,0)$-slice ($\Delta t = 2$)')
    else:
        plt.suptitle(r'Relative Errors for Poloidal Advection Step on $(z,v) = (0,0)$-slice')

    if save_plot:
        plt.savefig(foldername + 'rel_err_z0v0.png')

    if show_plot:
        plt.show()

    plt.close()

    # ==================================================
    # ======== Plot relative errors (log scale) ========
    # ==================================================
    # totally integrated
    plt.figure(figsize=(9.5, 7), dpi=250)
    p = 1
    for i in range(entries[0]):
        if labels[0][i] == 'l2_phi':
            continue
        else:
            plt.subplot(2, 2, p)
            p += 1
            for k in range(len(methods)):
                plt.plot(times[k], np.abs(np.divide(data[k][:, i] - data[k][:, i + 2*entries[k]], data[k][:, i])),
                         label=plot_labels[k], linestyle=styles[k])
            plt.legend()
        plt.subplots_adjust(left=0.05, right=0.99)
        plt.title(labels[k][i])
        plt.yscale('log')

    if dt[0] == 2:
        plt.suptitle(
            r'Relative Errors for Poloidal Advection Step ($\Delta t = 2$)')
    else:
        plt.suptitle(r'Relative Errors for Poloidal Advection Step')

    if save_plot:
        plt.savefig(foldername + 'rel_err_log.png')

    if show_plot:
        plt.show()

    plt.close()

    # (z,v) = (0,0) slice
    plt.figure(figsize=(9.5, 7), dpi=250)
    p = 1
    for i in range(entries[0]):
        if labels[0][i] == 'l2_phi':
            continue
        else:
            plt.subplot(2, 2, p)
            p += 1
            for k in range(len(methods)):
                plt.plot(times[k], np.abs(np.divide(data[k][:, i + entries[k]] - data[k][:, i + 3*entries[k]], data[k][:, i + entries[k]])),
                         label=plot_labels[k], linestyle=styles[k])
            plt.legend()
        plt.subplots_adjust(left=0.05, right=0.99)
        plt.title(labels[k][i])
        plt.yscale('log')

    if dt[0] == 2:
        plt.suptitle(
            r'Relative Errors for Poloidal Advection Step on $(z,v) = (0,0)$-slice ($\Delta t = 2$)')
    else:
        plt.suptitle(r'Relative Errors for Poloidal Advection Step on $(z,v) = (0,0)$-slice')

    if save_plot:
        plt.savefig(foldername + 'rel_err_z0v0_log.png')

    if show_plot:
        plt.show()

    plt.close()

    # ======================================
    # ======== Plot absolute errors ========
    # ======================================
    # totally integrated
    plt.figure(figsize=(9.5, 7), dpi=250)
    p = 1
    for i in range(entries[0]):
        if labels[0][i] == 'l2_phi':
            continue
        else:
            plt.subplot(2, 2, p)
            p += 1
            for k in range(len(methods)):
                plt.plot(times[k], np.abs(data[k][:, i] - data[k][:, i + 2*entries[k]]),
                         label=plot_labels[k], linestyle=styles[k])
            plt.legend()
        plt.subplots_adjust(left=0.05, right=0.99)
        plt.title(labels[k][i])
        if labels[k][i] == 'Kinetic Energy':
            plt.yscale('log')

    if dt[0] == 2:
        plt.suptitle(
            r'Absolute Errors for Poloidal Advection Step ($\Delta t = 2$)')
    else:
        plt.suptitle(r'Absolute Errors for Poloidal Advection Step')

    if save_plot:
        plt.savefig(foldername + 'abs_err.png')

    if show_plot:
        plt.show()

    plt.close()

    # (z,v) = (0,0) slice
    plt.figure(figsize=(9.5, 7), dpi=250)
    p = 1
    for i in range(entries[0]):
        if labels[0][i] == 'l2_phi':
            continue
        else:
            plt.subplot(2, 2, p)
            p += 1
            for k in range(len(methods)):
                plt.plot(times[k], np.abs(np.divide(data[k][:, i + entries[k]] - data[k][:, i + 3*entries[k]], data[k][:, i + entries[k]])),
                         label=plot_labels[k], linestyle=styles[k])
            plt.legend()
        plt.subplots_adjust(left=0.05, right=0.99)
        plt.title(labels[k][i])
        if labels[k][i] == 'Kinetic Energy':
            plt.yscale('log')

    if dt[0] == 2:
        plt.suptitle(
            r'Absolute Errors for Poloidal Advection Step on $(z,v) = (0,0)$-slice ($\Delta t = 2$)')
    else:
        plt.suptitle(r'Absolute Errors for Poloidal Advection Step on $(z,v) = (0,0)$-slice')

    if save_plot:
        plt.savefig(foldername + 'abs_err_z0v0.png')

    if show_plot:
        plt.show()

    plt.close()

    # ==================================================
    # ======== Plot absolute errors (log scale) ========
    # ==================================================
    # totally integrated
    plt.figure(figsize=(9.5, 7), dpi=250)
    p = 1
    for i in range(entries[0]):
        if labels[0][i] == 'l2_phi':
            continue
        else:
            plt.subplot(2, 2, p)
            p += 1
            for k in range(len(methods)):
                plt.plot(times[k], np.abs(data[k][:, i] - data[k][:, i + 2*entries[k]]),
                         label=plot_labels[k], linestyle=styles[k])
            plt.legend()
        plt.subplots_adjust(left=0.05, right=0.99)
        plt.title(labels[k][i])
        plt.yscale('log')

    if dt[0] == 2:
        plt.suptitle(
            r'Absolute Errors for Poloidal Advection Step ($\Delta t = 2$)')
    else:
        plt.suptitle(r'Absolute Errors for Poloidal Advection Step')

    if save_plot:
        plt.savefig(foldername + 'abs_err_log.png')

    if show_plot:
        plt.show()

    plt.close()

    # (z,v) = (0,0) slice
    plt.figure(figsize=(9.5, 7), dpi=250)
    p = 1
    for i in range(entries[0]):
        if labels[0][i] == 'l2_phi':
            continue
        else:
            plt.subplot(2, 2, p)
            p += 1
            for k in range(len(methods)):
                plt.plot(times[k], np.abs(np.divide(data[k][:, i + entries[k]] - data[k][:, i + 3*entries[k]], data[k][:, i + entries[k]])),
                         label=plot_labels[k], linestyle=styles[k])
            plt.legend()
        plt.subplots_adjust(left=0.05, right=0.99)
        plt.title(labels[k][i])
        plt.yscale('log')

    if dt[0] == 2:
        plt.suptitle(
            r'Absolute Errors for Poloidal Advection Step on $(z,v) = (0,0)$-slice ($\Delta t = 2$)')
    else:
        plt.suptitle(r'Absolute Errors for Poloidal Advection Step on $(z,v) = (0,0)$-slice')

    if save_plot:
        plt.savefig(foldername + 'abs_err_z0v0_log.png')

    if show_plot:
        plt.show()

    plt.close()


def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-cobra', metavar='foldername',
                        nargs='*', type=int, default=[0])
    parser.add_argument('-raven', metavar='foldername',
                        nargs='*', type=int, default=[0])
    parser.add_argument('--old', action='store_true')

    args = parser.parse_args()
    cobra = args.cobra[0]
    raven = args.raven[0]
    old = args.old

    k = 0

    parentpath = 'plotting/comparison/'
    if not os.path.exists(parentpath):
        os.mkdir(parentpath)

    while True:
        # folder_cobra = 'cobra/sim_' + str(cobra) + '/'
        folder_cobra = 'simulation_1/'
        # folder_cobra = 'raven/sim_' + str(cobra) + '/'
        folder_raven = 'simulation_0/'
        # folder_raven = 'raven/sim_' + str(raven) + '/'
        foldername = parentpath + str(k) + '/'
        if os.path.exists(foldername):
            k += 1
        else:
            os.mkdir(foldername)
            if old:
                plot_diagnostics_old(foldername,
                                     [folder_cobra, folder_raven],
                                     ['sl', 'akw'],
                                     #  ['akw', 'akw'],
                                     plot_labels=['Semi-Lagrangian', 'Arakawa'],
                                     #  plot_labels = [r'$\Delta t = 1$', r'$\Delta t = 2$'],
                                     styles=['solid', 'dashdot'])
            else:
                plot_diagnostics_new(foldername,
                                     [folder_cobra, folder_raven],
                                     ['sl', 'akw'],
                                     plot_labels=['Semi-Lagrangian', 'Arakawa'],
                                     styles=['solid', 'dashdot'])
            break


if __name__ == '__main__':
    main()
