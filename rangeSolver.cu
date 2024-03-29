#ifndef __CUDA_RUNTIME_H__
#include "cuda_runtime.h"
#endif // !"cuda_runtime.h"

#ifndef __DEVICE_LAUNCH_PARAMETERS_H__
#include "device_launch_parameters.h"
#endif // !__DEVICE_LAUNCH_PARAMETERS_H__

#include <stdio.h>
#include <string>
#include <iostream>
#include <chrono>
#include <ctime>
#include <vector>

#include "main.cuh"

using namespace std;

// Solves for a range of N: [lower, upper)
cudaError_t rangeSolve(int lower, int upper, int* cflag_ptr, int* board_ptr) {

	cudaError_t cudaStatus;

	// Solver status variables
	double tot_time = 0;
	vector<int> fail_list;
	int num_fails = 0;
	int pct_complete = 0;

	for (int Nq = lower; Nq < upper; Nq++) {

		// Allocate CUDA blocks and threads to dispatch
		int threadsPerBlock = 256;
		int blocksPerGrid = (Nq / 2 + threadsPerBlock) / threadsPerBlock;

		// GPU processing start time
		auto gpu_start = chrono::system_clock::now();

		// Execute main GPU code
		N_Queens_Kernel <<<blocksPerGrid, threadsPerBlock>>> (Nq);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cout << "Failure at N = " << Nq << endl;
			fprintf(stderr, "N_Queens_Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			memPurge();
			return cudaStatus;
		}

		// Wait for all cores to terminate
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			cout << "Failure at N = " << Nq << endl;
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching N_Queens_Kernel!\n", cudaStatus);
			memPurge();
			return cudaStatus;
		}

		// Copy verification flag state to host
		short local_flag = 0;
		cudaStatus = cudaMemcpy(&local_flag, cflag_ptr, sizeof(short), cudaMemcpyDeviceToHost);

		// Add current N to list if it fails verification
		if (local_flag != 0) {
			fail_list.push_back(Nq);
			num_fails += 1;
		}

		// Give status update every 1/10th of range
		if ((Nq - lower) % (1 + (upper - lower) / 10) == 0) {
			pct_complete = 100 * (Nq - lower + 1) / (upper - lower);
			cout << pct_complete << "% complete, up to N = " << Nq << ". # of failures: " << num_fails << endl;
		}

		// Clear board and occupancy grid
		clearBuffers <<<blocksPerGrid, threadsPerBlock>>> (Nq);

		// Check for any errors launching the kernels
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "clearBuffers kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			memPurge();
			return cudaStatus;
		}

		// Wait for all cores to terminate
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching clearBuffers!\n", cudaStatus);
			memPurge();
			return cudaStatus;
		}

		auto gpu_end = chrono::system_clock::now();
		chrono::duration<double> gpu_mseconds = (gpu_end - gpu_start) * 1000;
		tot_time += gpu_mseconds.count();

	}

	// Check if whole range passed
	if (num_fails == 0) {
		cout << "All solutions verified for range [" << lower << ", " << upper << "]." << endl;
	}
	cout << endl;
	cout << "Total GPU time (s): " << tot_time / 1000 << endl;

	return cudaStatus;
}
