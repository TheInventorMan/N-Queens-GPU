
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <ctime>

cudaError_t N_Queens();

// Global variables
const int Nq = (2147483648/8); //N = 1/2 maxint32 = 1,073,741,824 queens

// GPU-local variables
__device__ int board[Nq] = { 0 };
__device__ short occ_col[Nq]; // = { 0 };
__device__ short occ_row[Nq]; // = { 0 };
__device__ short occ_adiag[2 * Nq]; // = { 0 };
__device__ short occ_ddiag[2 * Nq]; // = { 0 };

__device__ short collision_flag[1] = { 0 };

// CUDA kernels
__device__ void register_q(int x, int y, int num_queens) {
	if (occ_col[x] != 0 || occ_row[y] != 0 || occ_adiag[(x + y)] != 0 || occ_ddiag[num_queens + (x - y)] != 0) {
		collision_flag[0] = 1;
	}
	occ_col[x] = 1;
	occ_row[y] = 1;
	occ_adiag[x + y] = 1;
	occ_ddiag[num_queens + (x - y)] = 1;
}

__global__ void N_Queens_Kernel(int num_queens)
{
	int i = (blockDim.x * blockIdx.x + threadIdx.x) + 1;
	int x, y, x1, y1;

	if (num_queens % 2 == 0 && (num_queens - 2) % 6 != 0) { // Case 1
		x = i - 1;
		y = 2 * i - 1;
		x1 = num_queens / 2 + i - 1;
		y1 = 2 * i - 2;

		register_q(x, y, num_queens);
		register_q(x1, y1, num_queens);

		board[x] = y;
		board[x1] = y1;
	}
	else if (num_queens % 2 == 0 && num_queens % 6 != 0) { // Case 2
		x = i - 1;
		y = (2 * i + num_queens / 2 - 3 % num_queens) % num_queens;
		x1 = num_queens - i;
		y1 = num_queens - (2 * i + num_queens / 2 - 3 % num_queens) - 1;
		if (y1 < 0) {
			y1 += num_queens;
		}

		register_q(x, y, num_queens);
		register_q(x1, y1, num_queens);

		board[x] = y;
		board[x1] = y1;
	}
	else {
		x = i - 1;
		y = 2 * i - 1;
		x1 = (num_queens-1) / 2 + i - 1;
		y1 = 2 * i - 2;

		register_q(x, y, num_queens-1);
		register_q(x1, y1, num_queens-1);

		board[x] = y;
		board[x1] = y1;

		//?__syncthreads();

		if (collision_flag[0] == 1 || occ_ddiag[0] == 1) {
			x = i - 1;
			y = (2 * i + num_queens / 2 - 3 % num_queens) % num_queens;
			x1 = num_queens - 1 - i;
			y1 = num_queens - 1 - (2 * i + (num_queens-1) / 2 - 3 % (num_queens-1)) - 1;
			if (y1 < 0) {
				y1 += num_queens-1;
			}

			register_q(x, y, num_queens-1);
			register_q(x1, y1, num_queens-1);

			board[x] = y;
			board[x1] = y1;
		}
		if (blockIdx.x == 0 && threadIdx.x == 0) {
			board[num_queens - 1] = num_queens - 1;
		}
		
	}

}

int main()
{
	auto global_start = std::chrono::system_clock::now();

    cudaError_t cudaStatus = N_Queens();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "queens died :(");
        return 1;
    }

	auto global_end = std::chrono::system_clock::now();	
	std::chrono::duration<double> elapsed_seconds = global_end - global_start;
	std::cout << "Total exec time (s): " << elapsed_seconds.count() << std::endl;
    

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t N_Queens() // (int* board, const int iter_size, unsigned int size) 
{
	short local_flag = 0;
	int loc_board[1];

    cudaError_t cudaStatus;
	int* cflag_ptr = 0;
	cudaStatus = cudaGetSymbolAddress((void**)&cflag_ptr, collision_flag);

	int* board_ptr = 0;
	cudaStatus = cudaGetSymbolAddress((void**)&board_ptr, board);
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	int threadsPerBlock = 256;
	int blocksPerGrid = (Nq/2 + threadsPerBlock - 1) / threadsPerBlock;

	auto gpu_start = std::chrono::system_clock::now();

	N_Queens_Kernel<< <blocksPerGrid, threadsPerBlock >> > (Nq);	

    // Check for any errors launching the kernels
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	cudaStatus = cudaMemcpy(&local_flag, cflag_ptr, sizeof(short), cudaMemcpyDeviceToHost);
	std::cout << "N = " << Nq << std::endl;
	if (local_flag == 0) {
		std::cout << "Solution verified" << std::endl;
	}


	auto gpu_end = std::chrono::system_clock::now();
	std::chrono::duration<double> gpu_mseconds = (gpu_end - gpu_start)*1000;
	std::cout << "GPU time (ms): " << gpu_mseconds.count() << std::endl;


    // Copy output vector from GPU buffer to host memory.
    //cudaStatus = cudaMemcpy(loc_board, board_ptr, Nq * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	//for (auto x : loc_board) {
	//	std::cout << x << std::endl;
	//}

Error:
    cudaFree(board);
	cudaFree(&collision_flag);
	cudaFree(occ_col);
	cudaFree(occ_row);
	cudaFree(occ_adiag);
	cudaFree(occ_ddiag);

    
    return cudaStatus; 
}
