
/**
 * Demo Program to mimic challenging large datasets use in HDF5
 * Problem size is pertinent to WARPM p.d.f. variable reads & writes for 3D+3V phase space
 **/

#include <mpi.h>
#include <hdf5.h>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <list>
#include <sstream>
#include <chrono>
#include <ctime>

// Dataset Shape that are not set at runtime
const int NV = 20;
const int dof = 729;
const int num_species = 2;

const char* output_filename_base = "pdf_out_";
//const char* input_filename  = "simulation_checkpoint_state.h5";
const char* COMPRESS_ARG = "COMPRESS";

std::list<unsigned> getPrimeFactors(unsigned n);
size_t setProblemSize( const int myrank, int numprocs, const char * const*tileStrings, size_t* tile_x, size_t* tile_y, size_t* tile_z, size_t* p_x, size_t* p_y, size_t* p_z, size_t* ipx, size_t* ipy, size_t* ipz  );
void initMemory( int tag, size_t memCount, double * local_data_ptr );



int
main(int argc, char **argv)
{
	//initialize_MPI_related:
	int mpi_thread_support_required = MPI_THREAD_FUNNELED;
	int mpi_thread_support_provided;
	int dummy_argc = 0;
	int result = MPI_Init_thread(&dummy_argc, NULL, mpi_thread_support_required, &mpi_thread_support_provided);


	int mympirank, numprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &mympirank);
	
	
	
	// HDF5 io object will be created using a clone of the MPI Communicator to reduce dead-lock likelihood with the ol' Global Arrays
	MPI_Comm cloned_world;
	result = MPI_Comm_dup(MPI_COMM_WORLD, &cloned_world);
	if (result != MPI_SUCCESS)
		throw std::runtime_error("MPI error duplicating MPI_COMM_WORLD.");
	
	
	// Dataset Shape
	size_t tile_x, tile_y, tile_z, p_x, p_y, p_z, ipx, ipy, ipz, globalArrayBytes;
	bool compress = false;
	try {
		if (argc < 4 )
			throw std::runtime_error("Too few arguments supplied.");
		globalArrayBytes = setProblemSize( mympirank, numprocs, argv+1, &tile_x, &tile_y, &tile_z, &p_x, &p_y, &p_z, &ipx, &ipy, &ipz );
		
		if (argc == 5) {
			if (std::string(argv[4]) == COMPRESS_ARG )
				compress = true;
			else
				throw std::runtime_error("Bad COMPRESS argument.");
		}
		if (argc > 5)
			std::runtime_error("Too many arguments supplied.");
	} catch(...) {
		std::cerr << "Check arguments. Expecting three integer arguments for process tile size: nx ny nz    and optional " << COMPRESS_ARG << std::endl;
		return 1;
	}
	
	
	constexpr unsigned rank = 8;
	hsize_t fileExtent[rank] = { tile_x*p_x, tile_y*p_y, tile_z*p_z, NV, NV, NV, dof, num_species };
	// Create the data region to be written
	hsize_t localExtent[rank] =  { tile_x, tile_y, tile_z, NV, NV, NV, dof, num_species }; // this would have ghost padding added
	hsize_t localStartOffset[rank] = {0,0,0,0,0,0,0,0}; // TODO: ghost padding in xyz here. // {[0] = 1, [1] = 1, [2] = 1, [3] = 0, [4] = 0, [5] = 0, [6] = 0, [7] = 0}
	hsize_t localCount[rank]  =  { tile_x, tile_y, tile_z, NV, NV, NV, dof, num_species }; // this one without ghost padding added
	hsize_t fileStartOffset[rank] = {tile_x*ipx, tile_y*ipy, tile_z*ipz, 0,0,0,0,0};; // {[0] = 0, [1] = 3, [2] = 0, [3] = 0, [4] = 0, [5] = 0, [6] = 0, [7] = 0}
	
	size_t memCount = tile_x * tile_y * tile_z; memCount *= NV * NV * NV; memCount *= dof * num_species;
	double* local_data_ptr = new double[memCount];
	initMemory( mympirank, memCount, local_data_ptr );
	std::stringstream output_filename_ss;
	output_filename_ss << output_filename_base << tile_x*p_x << "x" << tile_y*p_y << "x" << tile_z*p_z << (compress? "_compressed" : "" )<< ".h5";
	
	
	hid_t plistId = H5Pcreate(H5P_FILE_ACCESS);
	herr_t res = H5Pset_fapl_mpio(plistId, cloned_world, MPI_INFO_NULL);
	if (res < 0)
		throw std::runtime_error("WxHdf5Io::createFile: Internal Failure of H5Pset_fapl_mpio.");
	if (mympirank == 0) {
		std::time_t tt = std::chrono::system_clock::to_time_t( std::chrono::system_clock::now() );
		std::cout << "Beginning Time Critical HDF5 Dataset Write at " 
		          << std::ctime( & tt )
		          << std::endl << std::flush;
	}
	
	MPI_Barrier(MPI_COMM_WORLD); // focus any timing on the write phase	
	///////// BEGIN TIME CRITICAL REGION /////////////////////
	auto clock_start = std::chrono::system_clock::now();
	
	hid_t fn = H5Fcreate(output_filename_ss.str().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plistId);
	if (fn < 0)
		throw std::runtime_error("WxHdf5Io::createFile: unable to create file ");

	
	H5Pclose(plistId);
	
	
		
	
	hid_t variables_node = H5Gcreate( fn, "variables", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);				 
	//... omitted dozens of groups and attributes supporting checkpoint
	
	hid_t filespace = H5Screate_simple(rank, fileExtent, NULL);  // NULL restricts max size to initial size
	
	hid_t data_create_pl = H5Pcreate(H5P_DATASET_CREATE);
	hid_t data_access_pl = H5Pcreate(H5P_DATASET_ACCESS);
	
	// This writeDataSet implementation has been written assuming the entire dataset is written at once (thus the H5Dcreate call below inside writeDataSet)
	// and no provision or promise for what should be expected in unwritten regions of the data set. (undefined)
	// In this case, we don't want to write fill values that are subsequently overwritten. Documentation for H5D_FILL_TIME_IFSET says,
	// "Write fill values to the dataset when storage space is allocated only if there is a user-defined fill value"
	// H5D_FILL_TIME_NEVER could be just as appropriate except
	// development might come around later and add a fill value concept and H5Pset_fill_value() call.
	herr_t pset_ret1 = H5Pset_fill_time(data_create_pl, H5D_FILL_TIME_NEVER );
	
	
	// Data Set Chunking:
	hsize_t chunkShape[rank] = {1,1,1,NV,NV,NV,dof,num_species};
	herr_t pset_ret2 = H5Pset_chunk (data_create_pl, rank, &chunkShape[0]);
	
	// Chunk cache:
	const size_t cache_size_Bytes = /*8 chunks of cache:*/ 8*NV*NV*NV*dof*num_species*sizeof(double);
	herr_t pset_ret4 = H5Pset_chunk_cache(data_access_pl, H5D_CHUNK_CACHE_NSLOTS_DEFAULT, cache_size_Bytes, 1.0 /*prempt fully written chunks*/ );
	
	
	if (pset_ret1 < 0 || pset_ret2 < 0 || pset_ret4 < 0 || data_access_pl == H5I_INVALID_HID)
		throw std::runtime_error("WxHdf5IoTmpl::writeDataSet: Unexpected dataset property set failed.");
	
	
	// Chunk Compression Filter:
	if (compress) {
		unsigned int filter_info; herr_t filter_ret = -1;
		if( H5Zfilter_avail(H5Z_FILTER_DEFLATE) > 0 )
			if( H5Zget_filter_info (H5Z_FILTER_DEFLATE, &filter_info) >= 0)
				if ( (filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED) && (filter_info & H5Z_FILTER_CONFIG_DECODE_ENABLED) )
				{
					filter_ret = H5Pset_deflate(data_create_pl, 6 );
				}
		if ( filter_ret<0 )
			throw std::runtime_error("WxHdf5IoTmpl::writeDataSet: Unexpected deflate filter setup failure.");
	}	
	
	hid_t dn = H5Dcreate(variables_node, "pdf", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, data_create_pl, data_access_pl);
	herr_t pset_ret5 = H5Pclose(data_access_pl);
	herr_t pset_ret3 = H5Pclose(data_create_pl);
	herr_t ret4 = H5Gclose(variables_node);
	if (pset_ret3 < 0 || ret4 < 0 || pset_ret5 < 0)
		throw std::runtime_error("WxHdf5IoTmpl::writeDataSet: Unexpected H5?close failed.");
	
	
	if (dn == H5I_INVALID_HID) {
		H5Sclose(filespace);
		throw std::runtime_error("WxHdf5IoTmpl::writeDataSet: H5Dcreate failed.");
	}
	
	
	
	
	
	hid_t memoryspace = H5Screate_simple(rank, localExtent, NULL);  // this represents the entire local contiguos allocation

	hid_t ret1 = H5Sselect_hyperslab(memoryspace, H5S_SELECT_SET, localStartOffset, NULL, localCount, NULL);
	// pair down the file space to that extent held by this process. localCount and count in filespace are the same
	hid_t ret2 = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, fileStartOffset, NULL, localCount, NULL);

	
	if (ret1 < 0 || ret2 < 0) {
		H5Sclose(memoryspace);
		H5Sclose(filespace);
		H5Dclose(dn);
		throw std::runtime_error("WxHdf5IoTmpl::writeDataSet: H5Sselect_hyperslab failed based on local patch dimensions.");
	}  
	
	
	// Set for collective I/O
	hid_t xferPropList = H5Pcreate(H5P_DATASET_XFER);
	ret1 = H5Pset_dxpl_mpio(xferPropList, H5FD_MPIO_INDEPENDENT ); // what's better?  H5FD_MPIO_INDEPENDENT or H5FD_MPIO_COLLECTIVE

	if (ret1 < 0) {
		H5Sclose(memoryspace);
		H5Sclose(filespace);
		H5Dclose(dn);
		throw std::runtime_error("WxHdf5IoTmpl::writeDataSet: H5Pset_dxpl_mpio failed.");
	}
	
	// Write the data
	ret1 = H5Dwrite(dn, H5T_NATIVE_DOUBLE, memoryspace, filespace, xferPropList, local_data_ptr );

	
	H5Pclose(xferPropList);
	H5Sclose(memoryspace);
	H5Sclose(filespace);
	H5Dclose(dn);
	H5Fclose(fn);
	
	
	///////// END TIME CRITICAL REGION /////////////////////
	MPI_Barrier(MPI_COMM_WORLD);
	auto clock_stop = std::chrono::system_clock::now();
	auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(clock_stop - clock_start);
	const auto elapsed_time_ms = elapsed_time.count();
	if (mympirank == 0)
		std::cout << "HDF5 dataset write complete. Durration " << elapsed_time_ms*1e-3 << " seconds.  Effective write BW "<< globalArrayBytes*1e-9 / elapsed_time_ms *1e3 << " GB/s" << std::endl;
	
	// free allocations
	delete[] local_data_ptr;
	
	MPI_Finalize();
	return 0;	
}


/**
 * Simple 3D box decomposition
 * @in myrank - this process rank
 * @in numprocs - number of processes over which to decompose
 * @in tileStrings - array[3] of character strings to convert to per-process tile dimensions
 * @out tile_x, tile_y, tile_z : tile size in each dimension x,y,z
 * @out p_x, p_y, p_z  : number of tiles (processes) in each dimension
 * @out ipx, ipy, ipz  : 3d index of myrank among [p_x, p_y, p_z] decomposition box
 * @return Total Global Array Size in Bytes
 **/
size_t setProblemSize( const int myrank, int numprocs, const char * const *tileStrings, size_t* tile_x, size_t* tile_y, size_t* tile_z, size_t* p_x, size_t* p_y, size_t* p_z, size_t* ipx, size_t* ipy, size_t* ipz  )
{
	std::stringstream sx(tileStrings[0]), sy(tileStrings[1]), sz(tileStrings[2]);
	sx >> *tile_x; sy >> *tile_y; sz >> *tile_z;
	
	auto rankCountPrimeFactors = getPrimeFactors(numprocs);
	*p_x = 1; *p_y = 1; *p_z = 1;
	// quick decomposition algorithm that should result in a cube-ish process decomposition when possible
	while (rankCountPrimeFactors.size()) {
		*p_x *= rankCountPrimeFactors.back(); rankCountPrimeFactors.pop_back();
		
		if (rankCountPrimeFactors.size())
		{	*p_y	*= rankCountPrimeFactors.back(); rankCountPrimeFactors.pop_back(); }
		if (rankCountPrimeFactors.size())
		{	*p_z	*= rankCountPrimeFactors.back(); rankCountPrimeFactors.pop_back(); }
	}

	//coordinates for myrank in the process decomposition box:
	size_t stride = (*p_y)*(*p_z);
	*ipx = myrank / stride;
	int rem = myrank % stride;
	stride = (*p_z);
	*ipy = rem / stride;
	*ipz = rem % stride;
	if ( (*ipx >= *p_x) || (*ipy >= *p_y) || (*ipz >= *p_z)  )
		throw std::runtime_error("something in decomposition math didn't work out.");
	
	size_t globalArrayBytes = sizeof(double) * *tile_x * *tile_y * *tile_z;
	       globalArrayBytes *= *p_x * *p_y * *p_z;
	       globalArrayBytes *= NV * NV * NV;
	       globalArrayBytes *= dof * num_species;
	if (myrank == 0)
		std::cout << "Dataset global decomposition of tiles (processes): { " << *p_x << ", " << *p_y << ", " << *p_z << "}\n"
		          << "Tile Size: { " << *tile_x << ", " << *tile_y << ", " << *tile_z << ", " << NV << ", " << NV << ", " << NV << ", " << dof << ", " << num_species <<  "}\n"
		          << "Total GlobalArray Size " <<  std::setw(4+7) << std::setprecision(4) << std::scientific << globalArrayBytes*1.0 << " Bytes  "<< std::endl;
	return globalArrayBytes;
}

// compute primes of n.  List return is sorted largest to smallest
// The factor 1 is NOT included.
std::list<unsigned> getPrimeFactors(unsigned n)
{
	std::list<unsigned> primes;
	// quick and dirty, several optimizations are possible but we just don't anticipate working with large n
	for (unsigned i=2; i <= n; i++)
	{
		while(n % i == 0)
		{
			n /= i; // "n" divided by "i" is now "num"
			primes.push_front(i);
		}
	}
	return primes;
}

void initMemory( int tag, size_t memCount, double * local_data_ptr )
{
	for (size_t i=0; i< memCount; i++)
		local_data_ptr[i] = tag;
}


