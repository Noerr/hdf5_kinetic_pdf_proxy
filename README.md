# hdf5_kinetic_pdf_proxy
Proxy mini app that mimics HDF5 IO pattern for kinetic probability density function (PDF) in 3D3V 


 - The mini app creates an HDF5 file containing a single dataset organized as a rank-8 double-precision array.
 - Each process creates a same-sized tile
 - Executing with more MPI processes creates a large overall dataset and file
 - The dimensions {0,1,2} of the dataset is controlled by tile size and a simple 3d box arrangement of MPI ranks
 - The Dimensions {3,4,5,6,7} are compile-time fixed set by the parameters `NV`, `dof`, and `num_species`. These values match pertinent use case and have the effect of very large contiguous regions of memory that match contiguous regions in the HDF5 API 

 
## Compile & Build Examples

```
clang++ -std=c++17 -o ./demo_hdf5_use_pdf_vars.exe  -I${HDF5_DIR:?}/include ./demo_hdf5_use_pdf_vars.cpp
```

```
# Cray Programming Environment - presumes cray-mpich loaded
HDF5_DIR=~/hdf5_1.13.2   # use a custom build, or use an older hdf5 provided by module cray-hdf5-parallel
CC -std=c++17 -o ./demo_hdf5_use_pdf_vars.exe -I${HDF5_DIR:?}/include -L${HDF5_DIR}/lib -Wl,-rpath,${HDF5_DIR}/lib  -lhdf5 ./demo_hdf5_use_pdf_vars.cpp
```


## Run

The program is launched with three arguments describing the per-process tile size in dims 0,1,2:
`$TILE_SIDE_LENGTH1 $TILE_SIDE_LENGTH2 $TILE_SIDE_LENGTH3`

Likely considerations:
 - Set desired Lustre stripe settings for the output directory.
 - Ensure sufficient memory available for holding each process's tile on a compute node.

The file HDF5_IO_TEST.slurm provides a run script example.


## TODO

 [ ] At this time, the vales written to the dataset are not representative of real PDF values.