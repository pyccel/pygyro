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
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=emily.bourne@tum.de
#
# Wall clock limit:
#SBATCH --time=04:00:00


module load mkl anaconda/3/5.0.0 impi mpi4py h5py-mpi

module list


# Run the program:
time srun python3 l2Test.py 10 -f timeTest > prog.out
