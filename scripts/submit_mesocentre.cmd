#!/bin/sh
# Job Name:
#SBATCH -J pygyro
# Queue (Partition):
#SBATCH -p skylake
# Number of nodes and MPI tasks per node:
#SBATCH -N 8
#SBATCH --ntasks-per-node=32
#SBATCH -A b114
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
# Initial working directory:
#SBATCH -D ./
#
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=
#
# Wall clock limit:
#SBATCH --time=10:00:00

source scripts/modules_mesocentre.txt

module list

source ../venv/bin/activate

# Run the program:
srun python3 fullSimulation.py 4000 36000 -f test_C_code -c testSetups/iota0.json

deactivate
