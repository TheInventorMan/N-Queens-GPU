#ifndef __CUDA_RUNTIME_H__
#include "cuda_runtime.h"
#endif // !"cuda_runtime.h"

#ifndef __DEVICE_LAUNCH_PARAMETERS_H__
#include "device_launch_parameters.h"
#endif // !__DEVICE_LAUNCH_PARAMETERS_H__

using namespace std;

// Forward declarations
__device__ void register_q(int x, int y, int num_queens);
__device__ void case1(int i, int N);
__device__ void case2(int i, int N);
__global__ void N_Queens_Kernel(int num_queens);
__global__ void clearBuffers(int num_queens);

void memPurge();

int* getBoardAddr();
int* getFlagAddr();
int getMaxN();

// Host code forward declarations
cudaError_t singleSolve(int Nq, int* cflag_ptr, int* board_ptr);
cudaError_t rangeSolve(int lower, int upper, int* cflag_ptr, int* board_ptr);
void cls();