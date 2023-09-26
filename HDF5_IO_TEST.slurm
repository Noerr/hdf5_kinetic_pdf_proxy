#!/usr/bin/env bash
# This script requires one argument [Executable] and the rest are optional to override defaults
# sbatch this_script [Executable] [TILE_SIDE_LENGTH1=3] [TILE_SIDE_LENGTH2=3] [TILE_SIDE_LENGTH3=3] [NUM_PE=4*SLURM_NNODES]
# Expected arguments to SLURM are: -J name -N #nodes
#    Begin SLURM directives
#SBATCH --exclusive


echo Initial pwd: $(pwd)

module load craype-accel-nvidia80 #reportedly needed at build-time for for MPICH_GPU_SUPPORT_ENABLED
export MPICH_GPU_SUPPORT_ENABLED=1 #runtime factor
export MPICH_VERSION_DISPLAY=1

module unload darshan # unfortunately darshan has been nothing but trouble over the years (usually bad interactions with hdf5)
###module load cray-hdf5-parallel
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

module list
set -ex


#H5FClose hang for ~8 or more ranks troubleshooting
export MPICH_MPIIO_AGGREGATOR_PLACEMENT_DISPLAY=1
#export MPICH_MPIIO_HINTS_DISPLAY=1
export MPICH_MPIIO_STATS=1
export MPICH_MPIIO_ABORT_ON_RW_ERROR=enable
#export MPICH_MPIIO_HINTS='*.h5:cb_nodes=1:romio_cb_write=disable' #enable'
#export MPICH_MPIIO_HINTS=cb_nodes=1:romio_cb_write=enable


EXE=${1:?}
TILE_SIDE_LENGTH1=${2:-3}
TILE_SIDE_LENGTH2=${3:-3}
TILE_SIDE_LENGTH3=${4:-3}

PE_PER_NODE=4
if [ $# -ge 5 ] ; then
  NUM_PE=$5
else
  NUM_PE=$(expr $PE_PER_NODE '*' $SLURM_NNODES )
fi


ulimit -c unlimited

date
$SRUN_LAUNCH_PREFIX \
srun --ntasks=$NUM_PE \
	--gpu-bind=none `#closest`  \
	--gpus-per-task=1 --hint=nomultithread \
	--ntasks-per-node=$PE_PER_NODE --cpus-per-task=8 --exclusive --cpu_bind=verbose,sockets \
	${EXE:?} \
	$TILE_SIDE_LENGTH1 $TILE_SIDE_LENGTH2 $TILE_SIDE_LENGTH3

date

