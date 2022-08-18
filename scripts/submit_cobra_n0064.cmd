#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J pygyro
# Queue (Partition):
#SBATCH --partition=n0064
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=40
#
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=frederik.schnack@ipp.mpg.de
#
# Wall clock limit:
#SBATCH --time=24:00:00

module purge
module load intel/21.5.0
module load anaconda/3/2021.11
module load impi/2021.5
module load mpi4py
module load h5py-mpi
module load gcc/11
module list


# Run the program:
srun python3 fullSimulation.py 9000 85000 -f simulation_1/  > prog.out
