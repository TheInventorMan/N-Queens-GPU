#ifndef __CUDA_RUNTIME_H__
#include "cuda_runtime.h"
#endif // !"cuda_runtime.h"

#ifndef __DEVICE_LAUNCH_PARAMETERS_H__
#include "device_launch_parameters.h"
#endif // !__DEVICE_LAUNCH_PARAMETERS_H__

// Global variables
const int MAX_N = (2147483648 / 8); // N = 1/8 maxint32 = (2147483648 / 8) = 268,435,456 queens

// GPU-local variables
__device__ int board[MAX_N] = { 0 };   // list of queen positions, where board[x] = y
__device__ short occ_col[MAX_N];       // column occupancy
__device__ short occ_row[MAX_N];       // row occupancy
__device__ short occ_adiag[2 * MAX_N]; // ascending diagonal occupancy
__device__ short occ_ddiag[2 * MAX_N]; // decending diagonal occupancy
__device__ short collision_flag[1] = { 0 }; // Flag raised if any 2 Queens can attack each other

using namespace std;

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

int* getBoardAddr() {
	int* board_ptr = 0;
	cudaGetSymbolAddress((void**)&board_ptr, board);
	return board_ptr;
}

int* getFlagAddr() {
	int* cflag_ptr = 0;
	cudaGetSymbolAddress((void**)&cflag_ptr, collision_flag);
	return cflag_ptr;
}

int getMaxN() {
	int N = MAX_N;
	return N;
}

void memPurge() {
	cudaFree(board);
	cudaFree(collision_flag);
	cudaFree(occ_col);
	cudaFree(occ_row);
	cudaFree(occ_adiag);
	cudaFree(occ_ddiag);
}