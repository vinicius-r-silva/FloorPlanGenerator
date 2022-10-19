#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include "helloWorld.h"

__global__ void printHelloGPU()
{
	printf("Hello World from the GPUn");
}

int Cuda_test::launchGPU()
{
	printHelloGPU<<<1, 1>>>();
	cudaDeviceSynchronize();
	
	// getchar();
	return 0;
}