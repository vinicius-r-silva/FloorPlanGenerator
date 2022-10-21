#include <cstdio>
#include <iostream>
#include <vector>
#include <stdint.h>
#include "combine.h"
#include "helper.h"
#include "../lib/cvHelper.h"

#define __SIZE_A 12		// n_a * 4
#define __SIZE_B 12		// n_b * 4
#define __SIZE_RES 24	// n_res * 4


	// const int num_a = 1024;
	// const int num_threads = 1024;
	// const int num_blocks = (qtd_b + num_threads -1) / num_threads;
	// dim3 grid(num_blocks, num_a, 12);
	// dim3 threads(num_threads, 1, 1);

__global__ 
void printHelloGPU(int16_t *d_a, int16_t *d_b, int16_t *d_res, const int qtd_a, const int qtd_b) {
	const int k = blockIdx.z + 1 + blockIdx.z/4;
	const int a_idx = blockIdx.y;
	const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int res_idx = ((a_idx * blockDim.z + blockIdx.z) * qtd_b + b_idx) * __SIZE_RES;

	if(b_idx >= qtd_b || a_idx >= qtd_a)
		return;
	
	const int srcConn = k & 0b11;
	const int dstConn = (k >> 2) & 0b11;

	int16_t a[__SIZE_A];
	int16_t b[__SIZE_B];
	for(int i = 0; i < __SIZE_A; i++){
		a[i] = d_a[a_idx*__SIZE_A + i];
	}
	for(int i = 0; i < __SIZE_B; i++){
		b[i] = d_b[b_idx*__SIZE_B + i];
	}
	
	int dstX = 0;
	int dstY = 0;
	if(dstConn == 0 || dstConn == 2)
		dstX = b[0];
	else 
		dstX = b[2];
		
	if(dstConn == 0 || dstConn == 1)
		dstY = b[1];
	else 
		dstY = b[3];

	int srcX = 0;
	int srcY = 0;
	if(srcConn == 0 || srcConn == 2)
		srcX = a[__SIZE_A - 4];
	else 
		srcX = a[__SIZE_A - 2];
		
	if(srcConn == 0 || srcConn == 1)
		srcY = a[__SIZE_A - 3];
	else 
		srcY = a[__SIZE_A - 1];

	const int diffX = srcX - dstX;
	const int diffY = srcY - dstY;
	for(int i = 0; i < __SIZE_B; i+=2){
		b[i] += diffX;
		b[i+1] += diffY;	
	}

	for(int i = 0; i < __SIZE_A; i++){
		d_res[res_idx + i] = a[i];
	}
	// for(int i = 0; i < __SIZE_B; i++){
	// 	d_res[res_idx + i + __SIZE_A] = b[i];
	// }

	printf("a_idx: %d,\tb_idx: %d,\tres_idx: %d,\tk: %d\n", a_idx, b_idx, res_idx, k);
	// printf("a_idx: %d,\tb_idx: %d,\tres_idx: %d,\tk: %d,\tsrcX: %d,\tsrcY: %d,\tdstX: %d,\tdstY: %d,\tdiffX: %d,\tdiffY: %d\n", a_idx, b_idx, res_idx, k, srcX, srcY, dstX, dstY, diffX, diffY);
}

int Cuda_Combine::launchGPU(const std::vector<int16_t>& a, const std::vector<int16_t>& b) {
	// const int n_a = __SIZE_A / 4;
	// const int n_b = __SIZE_B / 4;
	const int qtd_a = a.size() / __SIZE_A;
	const int qtd_b = b.size() / __SIZE_B;
	findCudaDevice();	

	const int num_a = 1024;
	const int aLayoutSize = sizeof(int16_t) * __SIZE_A;
	const int bLayoutSize = sizeof(int16_t) * __SIZE_B;
	const int resLayoutSize = sizeof(int16_t) * __SIZE_RES;
	const unsigned int mem_size_a = aLayoutSize * qtd_a;
	const unsigned int mem_size_b = bLayoutSize * qtd_b;
	const unsigned int mem_size_res = num_a * 12 * qtd_b * resLayoutSize; //resLayoutSize * 12 * qtd_a * qtd_b
	
	// allocate host memory
	int16_t *h_a = (int16_t *)(&a[0]);
	int16_t *h_b = (int16_t *)(&b[0]);
	int16_t *h_res = (int16_t *)malloc(mem_size_res);

	// allocate device memory
	int16_t *d_a, *d_b, *d_res;
	checkCudaErrors(cudaMalloc((void **)&d_a, mem_size_a));
	checkCudaErrors(cudaMalloc((void **)&d_b, mem_size_b));
	checkCudaErrors(cudaMalloc((void **)&d_res, mem_size_res));

	// copy host data to device
	checkCudaErrors(cudaMemcpy(d_a, h_a, mem_size_a, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_b, h_b, mem_size_b, cudaMemcpyHostToDevice));

	// setup execution parameters
	const int num_threads = 1024;
	const int num_blocks = (qtd_b + num_threads -1) / num_threads;
	// const int num_threads = 6;
	// const int num_blocks = 1;
	dim3 grid(num_blocks, num_a, 12);
	dim3 threads(num_threads, 1, 1);

	printHelloGPU<<<grid, threads>>>(d_a, d_b, d_res, qtd_a, qtd_b);
	// printHelloGPU<<<grid, threads>>>(d_a, d_b, d_res, 1, 6);
	cudaDeviceSynchronize();	

	std::cout << "a.size(): " << a.size() << ", b.size(): " << b.size() << std::endl;
	std::cout << "qtd_a: " << qtd_a << ", qtd_b: " << qtd_b << std::endl;

	std::cout << "num_threads: " << num_threads << ", num_blocks: " << num_blocks << std::endl;
	std::cout << "mem_size_a: " << mem_size_a << ", mem_size_b: " << mem_size_b << ", mem_size_res: " << mem_size_res << std::endl;
	std::cout << "mem_size_a (MB): " << ((float)mem_size_a)/1024.0/1024.0 << ", mem_size_b (MB): " << ((float)mem_size_b)/1024.0/1024.0 << ", mem_size_res (MB): " << ((float)mem_size_res)/1024.0/1024.0 << std::endl;

	// check if kernel execution generated and error
	getLastCudaError("Kernel execution failed");

	// copy results from device to host
	checkCudaErrors(cudaMemcpy(h_res, d_res, mem_size_res, cudaMemcpyDeviceToHost));

	// std::cout << "A: " << std::endl;
	// for(int i = 0; i < 1 * __SIZE_A; i+=__SIZE_A){
	// 	for(int j = 0; j < __SIZE_A; j++){
	// 		std::cout << h_a[i + j] << ", ";
	// 	}
	// 	std::cout << std::endl;
	// }

	// std::cout << std::endl << "B: " << std::endl;
	// for(int i = 0; i < 6 * __SIZE_B; i+=__SIZE_B){
	// 	for(int j = 0; j < __SIZE_B; j++){
	// 		std::cout << h_b[i + j] << ", ";
	// 	}
	// 	std::cout << std::endl;
	// }

	// std::cout << std::endl << "Res: " << std::endl;
	// for(int i = 0; i < 6 * __SIZE_RES; i+=__SIZE_RES){
	// 	for(int j = 0; j < __SIZE_RES; j++){
	// 		std::cout << h_res[i + j] << ", ";
	// 	}
	// 	std::cout << std::endl;
	// }

	// std::vector<int16_t> PtsX(__SIZE_RES/2, 0);
	// std::vector<int16_t> PtsY(__SIZE_RES/2, 0);
	// for(int i = 0; i < mem_size_res / __SIZE_RES; i+=__SIZE_RES){
	// 	for(int j = 0; j < __SIZE_RES; j+=2){
	// 		PtsX[j/2] = h_res[i + j];
	// 		PtsY[j/2] = h_res[i + j + 1];
	// 	}
	// 	CVHelper::showLayout(PtsX, PtsY);
	// }


	// cleanup memory
	checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaFree(d_b));
	checkCudaErrors(cudaFree(d_res));
	// free(h_a);
	// free(h_b);
	free(h_res);

	// getchar();
	return 0;
}