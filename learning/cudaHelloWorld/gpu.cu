#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdio>
#define __N 4

__global__ void square(int16_t* in, int16_t* out) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int16_t res[4];
    res[0] = in[0];
    res[1] = in[1];
    res[2] = in[2];
    res[3] = in[3];

    out[0] = res[0];
    out[1] = res[1];
    out[2] = res[2];
    out[3] = res[3];
}

void launchKernel(){	
	int16_t *h_in = (int16_t*)calloc(__N, sizeof(int16_t));
	int16_t *h_out = (int16_t*)calloc(__N, sizeof(int16_t));

	for(int i = 0; i < __N; i++){
		h_in[i] = i;
	}

	const int mem_size = sizeof(int16_t) * __N;

	int16_t *d_in, *d_out;
	cudaMalloc((void **)&d_in, mem_size);
	cudaMalloc((void **)&d_out, mem_size);

	cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
	square<<<1, 1>>>(d_in, d_out);
	
	cudaDeviceSynchronize();

	cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

	// cleanup device memory
	cudaFree(d_in);
	cudaFree(d_out);

	for(int i = 0; i < __N; i++){
		std::cout << h_out[i] << ", ";
	}

	free(h_in);
	free(h_out);
}