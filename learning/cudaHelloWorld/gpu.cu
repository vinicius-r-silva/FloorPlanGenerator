#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>

__global__ void printHelloGPU()
{
	printf("Hello World from the GPU\n");
}

void launchKernel(){	
	printHelloGPU<<<1, 1>>>();
	cudaDeviceSynchronize();
}