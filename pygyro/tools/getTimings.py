import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Condense results')
parser.add_argument('nprocs', metavar='nprocs', nargs=1, type=int,
                    help='Number of processes used')

args = parser.parse_args()
nprocs = args.nprocs[0]

loop = np.empty(nprocs)
output = np.empty(nprocs)
setup = np.empty(nprocs)
additional = np.empty(nprocs)

for i in range(nprocs):
    file_object = open("timing/{}_l2Test{}.txt".format(nprocs, i), "r")
    loop[i] = float(file_object.read(19))
    output[i] = float(file_object.read(19))
    setup[i] = float(file_object.read(19))
    additional[i] = float(file_object.read(16))
    file_object.close()

loop_time = max(loop)
output_time = max(output)
setup_time = max(setup)
additional_time = max(additional)

file_object = open("results.txt", "a")
file_object.write("{nprocs:8}   {loop:16.10e}   {output:16.10e}   {setup:16.10e}   {additional:16.10e}\n"
                  .format(nprocs=nprocs, loop=loop_time, output=output_time, setup=setup_time, additional=additional_time))
