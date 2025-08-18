import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Condense results')
parser.add_argument('nprocs', metavar='nprocs', nargs=1, type=int,
                    help='Number of processes used')

args = parser.parse_args()
nprocs = args.nprocs[0]

steps = np.empty((nprocs, 5))
loop = np.empty(nprocs)
output = np.empty(nprocs)
setup = np.empty(nprocs)
additional = np.empty(nprocs)

for i in range(nprocs):
    with open(f"timing/{nprocs}_l2Test{i}.txt", "r") as file_object:
        for j in range(5):
            steps[i,j] = float(file_object.read(19))
        loop[i] = float(file_object.read(19))
        output[i] = float(file_object.read(19))
        setup[i] = float(file_object.read(19))
        additional[i] = float(file_object.read(16))

loop_time = max(loop)
output_time = max(output)
setup_time = max(setup)
additional_time = max(additional)

with open("results.txt", "a") as file_object:
    file_object.write(f"{nprocs:8}   ")
    for j in range(5):
        file_object.write(f"{max(steps[:,j]):8}   ")
    file_object.write(f"{loop_time:16.10e}   {output_time:16.10e}   {setup_time:16.10e}   {additional_time:16.10e}\n")
    #file_object.write(f"{nprocs:8}   {loop_time:16.10e}   {output_time:16.10e}   {setup_time:16.10e}   {additional_time:16.10e}\n")
