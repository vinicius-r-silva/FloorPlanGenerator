#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// #include <cstdio>
#include <stdio.h>
#include "helloWorld.h"

Cuda_test::Cuda_test(){
	printf("constructor\n");
}


int Cuda_test::launchGPU()
{
	printHelloGPU<<<1, 1>>>();
	cudaDeviceSynchronize();
	
	// getchar();
	return 0;
}

__host__ __device__ void Cuda_test::printHelloGPU()
{
	printf("Hello World from the GPUn");
}