#include <cstdio>
#include <iostream>
#include <vector>
#include <stdint.h>
#include "combine.h"
#include "helper.h"


__global__ void printHelloGPU()
{
	printf("Hello World from the GPU\n");
}

int Cuda_Combine::launchGPU(const std::vector<int>& a, const std::vector<int>& b, const int n_a, const int n_b) {
  std::cout << a.size() << b.size() << n_a << n_b << std::endl;
  CudaHelper::findCudaDevice();

  const unsigned int mem_size_a = sizeof(int) * a.size();
  const unsigned int mem_size_b = sizeof(int) * b.size();
  const unsigned int mem_size_res = sizeof(int) * 12 * a.size() * b.size();

	printHelloGPU<<<1, 1>>>();
	cudaDeviceSynchronize();
	
	// check if kernel execution generated and error
	CudaHelper::getLastCudaError("Kernel execution failed");
	// getchar();
	return 0;
}