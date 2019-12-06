#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
#
# Initial working directory:
#SBATCH -D ./
#
# Job Name:
#SBATCH -J pygyro
#
# Queue (Partition):
#SBATCH --account=FUA32_Selavlas
#SBATCH --partition=skl_fua_prod
#SBATCH --qos=skl_qos_fuabprod
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=32
#
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=yaman.guclu@ipp.mpg.de
#
# Wall clock limit:
#SBATCH --time=06:00:00

#==========================================
PYGYRO_DIR=~/Library/PyGyro
SIM_DIR=$CINECA_SCRATCH/timeTest_128nodes
#==========================================

# Load modules
module purge
source $PYGYRO_DIR/scripts/modules_marconi_intel.txt
module list

# Activate (local) Python virtual environment
source ~/PYGYRO/bin/activate

# Create simulation directory and move into it
mkdir "$SIM_DIR"
cd "$SIM_DIR"

# Copy PyGyro repository into simulation directory
cp -r "$PYGYRO_DIR" .

# Compile Fortran files
cd PyGyro; make COMP=intel MOPT=1; cd -

# Run program:
time srun python PyGyro/l2Test.py 6000 -f output -s 25 > prog.out
