#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J test_slurm
# Queue (Partition):
#SBATCH --partition=short
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#
#SBATCH --mail-type=none
#SBATCH --mail-user=emily.bourne@tum.de
#
# Wall clock limit:
#SBATCH --time=01:00:00

module load anaconda/3 impi mpi4py h5py-mpi

# Run the program:
srun python3 Pygyro/l2Test.py 100 -f timeTest > prog.out
