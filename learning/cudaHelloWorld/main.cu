#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdio>
#define __N_ARRAY 4
#define __SIZE_A 4

__global__ void square(int16_t* in, int16_t* out) {
    // int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int16_t res[__N_ARRAY];
	for(int i = 0; i < __N_ARRAY; i++){
		res[i] = in[i];
	}
	
	for(int i = 0; i < __N_ARRAY; i++){
		out[i] = res[i];
	}
}

__global__ 
void sharedTest(int16_t *d_in, int16_t *d_out){
	__shared__ int16_t a[__SIZE_A];
	if(threadIdx.x < __SIZE_A){
		a[threadIdx.x] = d_in[threadIdx.x];
	}
  	__syncthreads();

	const int idx = threadIdx.x * __SIZE_A;
	for(int i = 0; i < __SIZE_A; i++){
		d_out[idx + i] = a[i];
	}

	printf("threadIdx.x: %d,\tidx: %d\n", threadIdx.x, idx);
}

void launchSharedTest(){
	const int n_a = 2;
	const int in_mem_size = __SIZE_A * sizeof(int16_t);
	const int out_mem_size = n_a * __SIZE_A * __SIZE_A * sizeof(int16_t);

	int16_t *h_in = (int16_t*)malloc(in_mem_size);
	int16_t *h_out = (int16_t*)malloc(out_mem_size);

	for(int i = 0; i < __SIZE_A; i++){
		h_in[i] = i;
	}

	int16_t *d_in, *d_out;
	cudaMalloc((void **)&d_in, in_mem_size);
	cudaMalloc((void **)&d_out, out_mem_size);

	cudaMemcpy(d_in, h_in, in_mem_size, cudaMemcpyHostToDevice);

	dim3 grid(1, 1, 1);
	dim3 threads(__SIZE_A * n_a, 1, 1);
	sharedTest<<<grid, threads>>>(d_in, d_out);
	
	cudaDeviceSynchronize();

	cudaMemcpy(h_out, d_out, out_mem_size, cudaMemcpyDeviceToHost);

	// cleanup device memory
	cudaFree(d_in);
	cudaFree(d_out);

	for(int i = 0; i < n_a; i++){
		for(int j = 0; j < __SIZE_A; j++){
			for(int k = 0; k < __SIZE_A; k++){
				std::cout << h_out[i*__SIZE_A*__SIZE_A + j*__SIZE_A + k] << ", ";
			}
			std::cout << std::endl;
		}
	}

	free(h_in);
	free(h_out);

}

void launchKernel(){	
	int16_t *h_in = (int16_t*)calloc(__N_ARRAY, sizeof(int16_t));
	int16_t *h_out = (int16_t*)calloc(__N_ARRAY, sizeof(int16_t));

	for(int i = 0; i < __N_ARRAY; i++){
		h_in[i] = i;
	}

	const int mem_size = sizeof(int16_t) * __N_ARRAY;

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

	for(int i = 0; i < __N_ARRAY; i++){
		std::cout << h_out[i] << ", ";
	}

	free(h_in);
	free(h_out);
}


int main()
{
	// launchKernel();
	launchSharedTest();
	// getchar();
	return 0;
}