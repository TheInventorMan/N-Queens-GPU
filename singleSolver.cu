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

cudaError_t singleSolve(int Nq, int* cflag_ptr, int* board_ptr) {

	short local_flag = 0;

	// Get pointers to GPU buffers
	cudaError_t cudaStatus;

	// Allocate CUDA blocks and threads to dispatch
	int threadsPerBlock = 256;
	int blocksPerGrid = (Nq / 2 + threadsPerBlock) / threadsPerBlock;

	cout << "Launching " << blocksPerGrid << " block with " << threadsPerBlock << " threads each." << endl;
	cout << endl;

	// Display case number depending on value of N
	if (Nq % 2 == 0 && (Nq - 2) % 6 != 0) {
		cout << "Computing... (Case 1)" << endl;
	}
	else if (Nq % 2 == 0 && Nq % 6 != 0) {
		cout << "Computing... (Case 2)" << endl;
	}
	else if ((Nq - 1) % 2 == 0 && (Nq - 3) % 6 != 0) {
		cout << "Computing... (Case 3)" << endl;
	}
	else if ((Nq - 1) % 2 == 0 && (Nq - 1) % 6 != 0) {
		cout << "Computing... (Case 4)" << endl;
	}
	cout << endl;

	auto gpu_start = chrono::system_clock::now(); // GPU processing start time

	N_Queens_Kernel << <blocksPerGrid, threadsPerBlock >> > (Nq); // Execute GPU code

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		memPurge();
		return cudaStatus;
	}

	// Wait for all cores to terminate
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching N_Queens_Kernel!\n", cudaStatus);
		memPurge();
		return cudaStatus;
	}

	// Copy verification flag state to host
	cudaStatus = cudaMemcpy(&local_flag, cflag_ptr, sizeof(short), cudaMemcpyDeviceToHost);

	// Verbose debug output
	cout << "N = " << Nq << endl;
	if (local_flag == 0) {
		cout << "Solution verified" << endl;
	}
	cout << endl;

	auto gpu_end = chrono::system_clock::now();
	chrono::duration<double> gpu_mseconds = (gpu_end - gpu_start) * 1000;
	cout << "GPU time (ms): " << gpu_mseconds.count() << endl;

	// Copy output vector from GPU buffer to host memory. Only works for N < 32 (arbitrary)
	if (Nq < 32) {
		int loc_board[32];
		cudaStatus = cudaMemcpy(loc_board, board_ptr, Nq * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			memPurge();
			return cudaStatus;
		}
		cout << endl;
		cout << "Solution: " << endl;

		for (int i = 0; i < Nq; i++) {
			for (int j = 0; j < Nq; j++) {
				if (j == loc_board[i]) {
					cout << "X" << " ";
				}
				else {
					cout << "-" << " ";
				}
			}
			cout << endl;
		}
		cout << endl;
	}
	else {
		double sol_size = Nq * 16;
		string suffix = " bytes.";
		string prefix = "";

		if (sol_size > 1000000000) {
			sol_size /= 1000000000;
			suffix = " GB.";
		}
		else if (sol_size > 1000000) {
			sol_size /= 1000000;
			suffix = " MB.";
		}
		else if (sol_size > 1000) {
			sol_size /= 1000;
			suffix = " KB.";
		}
		else if (sol_size == 0) {
			sol_size = 2.048;
			prefix = ">";
			suffix = " GB.";
		}
		cout << "Solution too large to display. Solution size: " << prefix << sol_size << suffix << endl;
	}
	cout << endl;

	// Clear board and occupancy grid
	clearBuffers << <blocksPerGrid, threadsPerBlock >> > (Nq);

	// Check for any errors launching the kernels
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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

	return cudaStatus;

}