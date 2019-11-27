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

void cls() {
	std::cout << "\033[2J\033[1;1H";
}

int main()
{
	cudaError_t cudaStatus;

	// Store pointers to GPU memory locally
	int* cflag_ptr = getFlagAddr();
	int* board_ptr = getBoardAddr();

	int MAX_N = getMaxN();
	
	// Initialize GPU
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	auto global_start = chrono::system_clock::now(); // Program start time

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

			int lower, upper;

			cout << "Enter lower bound: ";
			cin >> lower;
			cout << "Enter upper bound: ";
			cin >> upper;

			cls();

			global_start = chrono::system_clock::now();

			cudaStatus = rangeSolve(lower, upper, cflag_ptr, board_ptr);

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

			cudaStatus = singleSolve(Nq, cflag_ptr, board_ptr);

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