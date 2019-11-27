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

// Global variables
const int MAX_N = (2147483648 / 8); // N = 1/8 maxint32 = (2147483648 / 8) = 268,435,456 queens

// GPU-local variables
__device__ int board[MAX_N] = { 0 };   // list of queen positions, where board[x] = y
__device__ short occ_col[MAX_N];       // column occupancy
__device__ short occ_row[MAX_N];       // row occupancy
__device__ short occ_adiag[2 * MAX_N]; // ascending diagonal occupancy
__device__ short occ_ddiag[2 * MAX_N]; // decending diagonal occupancy
__device__ short collision_flag[1] = { 0 }; // Flag raised if any 2 Queens can attack each other

// Forward declarations
__device__ void register_q(int x, int y, int num_queens);
__device__ void case1(int i, int N);
__device__ void case2(int i, int N);
__global__ void N_Queens_Kernel(int num_queens);
__global__ void clearBuffers(int num_queens);

// GPU functions
__device__ void register_q(int x, int y, int num_queens) // Check for collision and add queen to occupancy lists
{
	if (occ_col[x] != 0 || occ_row[y] != 0 || occ_adiag[(x + y)] != 0 || occ_ddiag[num_queens + (x - y)] != 0) {
		collision_flag[0] = 1;
	}

	occ_col[x] = 1;
	occ_row[y] = 1;
	occ_adiag[x + y] = 1;
	occ_ddiag[num_queens + (x - y)] = 1;

	return;
}

__device__ void case1(int i, int N) {
	int x, y, x1, y1;
	x = i;
	y = 2 * i;
	x1 = N / 2 + i;
	y1 = 2 * i - 1;

	register_q(x - 1, y - 1, N);
	register_q(x1 - 1, y1 - 1, N);

	board[x - 1] = y - 1;
	board[x1 - 1] = y1 - 1;

	return;
}

__device__ void case2(int i, int N) {
	int x, y, x1, y1;
	x = i;
	y = 1 + ((2 * (i - 1) + N / 2 - 1) % N);
	x1 = N + 1 - i;
	y1 = N - ((2 * (i - 1) + N / 2 - 1) % N);

	register_q(x - 1, y - 1, N);
	register_q(x1 - 1, y1 - 1, N);

	board[x - 1] = y - 1;
	board[x1 - 1] = y1 - 1;

	return;
}

// GPU kernel
__global__ void N_Queens_Kernel(int num_queens)
{

	int i = (blockDim.x * blockIdx.x + threadIdx.x) + 1; // Each thread places 2 queens

	if (i > (num_queens - num_queens % 2) / 2) {
		return;
	}

	if (num_queens % 2 == 0 && (num_queens - 2) % 6 != 0) { // Case 1, N is even and (N-2) mod 6 is not 0
		case1(i, num_queens);
	}
	else if (num_queens % 2 == 0 && num_queens % 6 != 0) { // Case 2, N is even and N mod 6 is not 0
		case2(i, num_queens);
	}
	else if ((num_queens - 1) % 2 == 0 && (num_queens - 3) % 6 != 0) { // Case 3, N is odd, and (N-3) mod 6 is not 0
		case1(i, num_queens - 1);
		if (blockIdx.x == 0 && threadIdx.x == 0) {
			board[num_queens - 1] = num_queens - 1;
		}
	}
	else if ((num_queens - 1) % 2 == 0 && (num_queens - 1) % 6 != 0) { // Case 4, N is odd and (N-1) mod 6 is not 0
		case2(i, num_queens - 1);
		if (blockIdx.x == 0 && threadIdx.x == 0) {
			board[num_queens - 1] = num_queens - 1;
		}
	}

	return;
}

__global__ void clearBuffers(int num_queens) {
	int i = (blockDim.x * blockIdx.x + threadIdx.x); // i < n/2

	board[2 * i] = 0;
	board[2 * i + 1] = 0;

	occ_col[2 * i] = 0;
	occ_col[2 * i + 1] = 0;

	occ_row[2 * i] = 0;
	occ_row[2 * i + 1] = 0;

	occ_adiag[2 * i] = 0;
	occ_adiag[2 * i + 1] = 0;
	occ_adiag[2 * i + num_queens] = 0;
	occ_adiag[2 * i + 1 + num_queens] = 0;

	occ_ddiag[2 * i] = 0;
	occ_ddiag[2 * i + 1] = 0;
	occ_ddiag[2 * i + num_queens] = 0;
	occ_ddiag[2 * i + 1 + num_queens] = 0;

	if (blockIdx.x == 0 && threadIdx.x == 0) {
		collision_flag[0] = 0;
	}

}

// Host code forward declarations
cudaError_t singleSolve(int Nq);
cudaError_t rangeSolve(int lower, int upper);
void memPurge();
void cls();


using namespace std;

void cls() {
	std::cout << "\033[2J\033[1;1H";
}

void memPurge() {
	cudaFree(board);
	cudaFree(collision_flag);
	cudaFree(occ_col);
	cudaFree(occ_row);
	cudaFree(occ_adiag);
	cudaFree(occ_ddiag);
}

cudaError_t rangeSolve(int lower, int upper) {
	using namespace std;

	// Store pointers to GPU memory locally
	int* cflag_ptr = 0;
	int* board_ptr = 0;
	short local_flag = 0;

	// Get pointers to GPU buffers
	cudaError_t cudaStatus;
	cudaStatus = cudaGetSymbolAddress((void**)&cflag_ptr, collision_flag);
	cudaStatus = cudaGetSymbolAddress((void**)&board_ptr, board);

	double tot_time = 0;
	vector<int> fail_list;
	int num_fails = 0;
	int pct_complete = 0;

	for (int Nq = lower; Nq < upper; Nq++) {

		// Allocate CUDA blocks and threads to dispatch
		int threadsPerBlock = 256;
		int blocksPerGrid = (Nq / 2 + threadsPerBlock) / threadsPerBlock;

		auto gpu_start = chrono::system_clock::now(); // GPU processing start time

		N_Queens_Kernel << <blocksPerGrid, threadsPerBlock >> > (Nq); // Execute GPU code

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cout << "Failure at N = " << Nq << endl;
			fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
		cudaStatus = cudaMemcpy(&local_flag, cflag_ptr, sizeof(short), cudaMemcpyDeviceToHost);

		if (local_flag != 0) {
			fail_list.push_back(Nq);
			num_fails += 1;
		}

		if ((Nq - lower) % (1+(upper-lower)/10) == 0) {
			pct_complete = 100 * (Nq - lower + 1) / (upper - lower);
			cout << pct_complete << "% complete, up to N = " << Nq << ". # of failures: " << num_fails << endl;
		}

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

		auto gpu_end = chrono::system_clock::now();
		chrono::duration<double> gpu_mseconds = (gpu_end - gpu_start) * 1000;
		tot_time += gpu_mseconds.count();

	}
	if (num_fails == 0) {
		cout << "All solutions verified for range [" << lower << ", " << upper << "]." << endl;
	}
	cout << endl;
	cout << "Total GPU time (s): " << tot_time/1000 << endl;
	return cudaStatus;
}

cudaError_t singleSolve(int Nq) {

	// Store pointers to GPU memory locally
	int* cflag_ptr = 0;
	int* board_ptr = 0;
	short local_flag = 0;

	// Get pointers to GPU buffers
	cudaError_t cudaStatus;
	cudaStatus = cudaGetSymbolAddress((void**)&cflag_ptr, collision_flag);
	cudaStatus = cudaGetSymbolAddress((void**)&board_ptr, board);

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

int main()
{
	using namespace std;
	cudaError_t cudaStatus;
	int* board_ptr = board;

	auto global_start = chrono::system_clock::now(); // Program start time

	// Initialize GPU
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	while (1) {

		cout << "Interactive GPU-Accelerated N-Queens Solver" << endl;
		cout << "Please select an option: " << endl;
		cout << "1 - Solve for N" << endl;
		cout << "2 - Solve for range of N" << endl;
		cout << "3 - Quit" << endl;

		int resp = 0;
		char _;

		cin >> resp;
		if (resp == 3) {
			break;
		}

		else if (resp == 2) {
			//cout << "Not implemented, press any key to continue." << endl;
			//cin >> _;
			//cls();
			//continue;

			int lower, upper;

			cout << "Enter lower bound: ";
			cin >> lower;
			cout << "Enter upper bound: ";
			cin >> upper;

			cls();

			global_start = chrono::system_clock::now();

			cudaStatus = rangeSolve(lower, upper);

			auto global_end = chrono::system_clock::now();
			chrono::duration<double> elapsed_seconds = (global_end - global_start);
			cout << "Total exec time (s): " << elapsed_seconds.count() << endl;

			cout << endl;
			cout << "Press any key to continue." << endl;
			cin >> _;
			cls();

		}
		else if (resp == 1) {

			int Nq = 0;

			cout << "Enter number of queens between 4 and " << MAX_N << ": ";
			cin >> Nq;

			if (Nq < 4 || Nq > MAX_N) {
				cls();
				continue;
			}

			cls();

			global_start = chrono::system_clock::now();

			cudaStatus = singleSolve(Nq);

			// Display total execution time
			auto global_end = chrono::system_clock::now();
			chrono::duration<double> elapsed_mseconds = 1000*(global_end - global_start);
			cout << "Total exec time (ms): " << elapsed_mseconds.count() << endl;

			cout << "Press any key to continue." << endl;
			cin >> _;
			cls();
		}
	}

	// Free up all GPU memory
Error:
	memPurge();

	// Ensure no errors on the status flag
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "queens died :(");
		return 1;
	}

	// cudaDeviceReset must be called before exiting
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}
