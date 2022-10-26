#include <cstdio>
#include <iostream>
#include <vector>
#include <stdint.h>

#include "combine.h"
#include "helper.h"
#include "../lib/cvHelper.h"
#include "../lib/globals.h"

#define __SIZE_A 12		// n_a * 4
#define __SIZE_B 12		// n_b * 4
#define __SIZE_RES 24	// n_res * 4


// const int num_threads = 768; // 1024
// const int num_blocks = (qtd_b + num_threads -1) / num_threads;
// dim3 grid(num_blocks, num_a, NConn);
// dim3 threads(num_threads, 1, 1);

// __global__ 
// void createPtsA(int16_t *d_a, int16_t *d_pts, const int qtd_a, const int qtd_b, const int a_offset) {
// 	const int k = blockIdx.z + 1 + blockIdx.z/4;
// 	const int a_idx = (blockIdx.y + a_offset) * __SIZE_A;
// 	const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
// 	int res_idx = ((blockIdx.z * qtd_a + blockIdx.y) * qtd_b +  b_idx) * __SIZE_RES;

// 	if(b_idx >= qtd_b || blockIdx.y >= qtd_a)
// 		return;

// 	// __shared__ int16_t a[__SIZE_A];
// 	// if(threadIdx.x < __SIZE_A){
// 	// 	a[threadIdx.x] = d_a[a_idx + threadIdx.x];
// 	// 	// printf("a_idx: %d,\tb_idx: %d,\tres_idx: %d,\tblockIdx.x: %d,\tblockIdx.y: %d,\tblockIdx.z: %d,\tdblockDim.x: %d,\tthreadIdx.x: %d\n", a_idx, b_idx, res_idx, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x);
// 	// }
//   	// __syncthreads();

// 	// for(int i = 0; i < __SIZE_A; i++){
// 	// 	d_pts[res_idx + i] = a[i];
// 	// }

// 	for(int i = 0; i < __SIZE_A; i++){
// 		d_pts[res_idx + i] = d_a[a_idx + i];
// 	}
// }

// __global__ 
// void createPtsB(int16_t *d_a, int16_t *d_b, int16_t *d_pts, const int qtd_a, const int qtd_b, const int a_offset) {
// 	const int k = blockIdx.z + 1 + blockIdx.z/4;
// 	const int a_idx = (blockIdx.y + a_offset) * __SIZE_A;
// 	int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
// 	int res_idx = ((blockIdx.z * qtd_a + blockIdx.y) * qtd_b +  b_idx) * __SIZE_RES;

// 	if(b_idx >= qtd_b || blockIdx.y >= qtd_a)
// 		return;

// 	b_idx *= __SIZE_B;
// 	int16_t b[__SIZE_B];
// 	for(int i = 0; i < __SIZE_B; i++){
// 		b[i] = d_b[b_idx + i];
// 	}
	
// 	const int srcConn = k & 0b11;
// 	const int dstConn = (k >> 2) & 0b11;
	
// 	int dst = 0;
// 	int src = 0;
// 	if(dstConn == 0 || dstConn == 2)
// 		dst = b[0];
// 	else 
// 		dst = b[2];

// 	if(srcConn == 0 || srcConn == 2)
// 		src = d_a[a_idx + __SIZE_A - 4];
// 	else 
// 		src = d_a[a_idx + __SIZE_A - 2];

// 	const int diffX = src - dst;
// 	for(int i = 0; i < __SIZE_B; i+=2){
// 		b[i] += diffX;
// 	}

// 	if(dstConn == 0 || dstConn == 1)
// 		dst = b[1];
// 	else 
// 		dst = b[3];
		
// 	if(srcConn == 0 || srcConn == 1)
// 		src = d_a[a_idx + __SIZE_A - 3];
// 	else 
// 		src = d_a[a_idx + __SIZE_A - 1];

// 	const int diffY = src - dst;
// 	for(int i = 1; i < __SIZE_B; i+=2){
// 		b[i] += diffY;
// 	}

// 	res_idx += __SIZE_A;
// 	for(int i = 0; i < __SIZE_B; i++){
// 		d_pts[res_idx + i] = b[i];
// 	}

// }

__global__ 
void createPts(int16_t *d_a, int16_t *d_b, int16_t *d_pts, const int qtd_a, const int qtd_b, const int a_offset) {
	const int k = blockIdx.z + 1 + blockIdx.z/4;
	const int a_idx = (blockIdx.y + a_offset) * __SIZE_A;
	const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int res_idx = ((blockIdx.z * qtd_a + blockIdx.y) * qtd_b +  b_idx) * __SIZE_RES;

	if(b_idx >= qtd_b || blockIdx.y >= qtd_a)
		return;


	__shared__ int16_t a[__SIZE_A];
	if(threadIdx.x < __SIZE_A){
		a[threadIdx.x] = d_a[a_idx + threadIdx.x];
		// printf("a_idx: %d,\tb_idx: %d,\tres_idx: %d,\tblockIdx.x: %d,\tblockIdx.y: %d,\tblockIdx.z: %d,\tdblockDim.x: %d,\tthreadIdx.x: %d\n", a_idx, b_idx, res_idx, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x);
	}
  	__syncthreads();

	// printf("a_idx: %d\tblock.x: %d\tblock.y: %d\tthread.x: %d\t%d %d %d %d %d %d %d %d %d %d %d %d\n", a_idx, blockIdx.x, blockIdx.y, threadIdx.x, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11]);
	// // int16_t res[__SIZE_A + __SIZE_B];
	// for(int i = 0; i < __SIZE_A; i++){
	// 	a[i] = d_a[a_idx + i];
	// }

	int16_t b[__SIZE_B];
	for(int i = 0; i < __SIZE_B; i++){
		b[i] = d_b[b_idx*__SIZE_B + i];
	}
	
	const int srcConn = k & 0b11;
	const int dstConn = (k >> 2) & 0b11;
	
	int dst = 0;
	int src = 0;
	if(dstConn == 0 || dstConn == 2)
		dst = b[0];
	else 
		dst = b[2];

	if(srcConn == 0 || srcConn == 2)
		src = a[__SIZE_A - 4];
	else 
		src = a[__SIZE_A - 2];

	const int diffX = src - dst;
	for(int i = 0; i < __SIZE_B; i+=2){
		b[i] += diffX;
	}

	if(dstConn == 0 || dstConn == 1)
		dst = b[1];
	else 
		dst = b[3];
		
	if(srcConn == 0 || srcConn == 1)
		src = a[__SIZE_A - 3];
	else 
		src = a[__SIZE_A - 1];

	const int diffY = src - dst;
	for(int i = 1; i < __SIZE_B; i+=2){
		b[i] += diffY;
	}

	for(int i = 0; i < __SIZE_A; i++){
		d_pts[res_idx + i] = a[i];
	}

	res_idx += __SIZE_A;
	for(int i = 0; i < __SIZE_B; i++){
		d_pts[res_idx + i] = b[i];
	}

	// for(int i = 0; i < __SIZE_A + __SIZE_B; i++){
	// 	d_pts[res_idx + i] = res[i];
	// }
		
	// if(dstConn == 0 || dstConn == 1)
	// 	dstY = b[1];
	// else 
	// 	dstY = b[3];

	// int srcX = 0;
	// int srcY = 0;
	// if(srcConn == 0 || srcConn == 2)
	// 	srcX = a[__SIZE_A - 4];
	// else 
	// 	srcX = a[__SIZE_A - 2];
		
	// if(srcConn == 0 || srcConn == 1)
	// 	srcY = a[__SIZE_A - 3];
	// else 
	// 	srcY = a[__SIZE_A - 1];

	// const int diffX = srcX - dstX;
	// const int diffY = srcY - dstY;
	// for(int i = 0; i < __SIZE_B; i+=2){
	// 	b[i] += diffX;
	// 	b[i+1] += diffY;	
	// }

	// for(int i = 0; i < __SIZE_A; i++){
	// 	d_pts[res_idx + i] = a[i];
	// }

	// res_idx += __SIZE_A;
	// for(int i = 0; i < __SIZE_B; i++){
	// 	d_pts[res_idx + i] = b[i];
	// }

	// if(res_idx > 80000000)
	// printf("k: %d,\ta_idx: %d,\tb_idx: %d,\tres_idx: %d\n", blockIdx.z, a_idx, b_idx, res_idx);
	// printf("a_idx: %d,\tb_idx: %d,\tres_idx: %d,\tblockIdx.x: %d,\tblockIdx.y: %d,\tblockIdx.z: %d,\tdblockDim.x: %d,\tthreadIdx.x: %d\n", a_idx, b_idx, res_idx - __SIZE_A, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x);
	// printf("a_idx: %d,\tb_idx: %d,\tres_idx: %d,\tblockIdx.x: %d,\tblockIdx.y: %d,\tblockIdx.z: %d,\tdblockDim.x: %d,\tthreadIdx.x: %d,\tdiffX: %d,\tdiffY: %d,\ta[0]: %d,\ta[1]: %d\n", a_idx, b_idx, res_idx - __SIZE_A, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, diffX, diffY, a[0], a[1]);
}

void Cuda_Combine::launchGPU(const std::vector<int16_t>& a, const std::vector<int16_t>& b) {
	const int NConn = 12;
	const long num_a = 170;
	const int qtd_a = a.size() / __SIZE_A;
	const int qtd_b = b.size() / __SIZE_B;

	// const long NConn = 2;
	// const long qtd_a = 2; 
	// const long qtd_b = 12; //minimum 12
	// const long num_a = qtd_a;

	findCudaDevice();	

	const long aLayoutSize = sizeof(int16_t) * __SIZE_A;
	const long bLayoutSize = sizeof(int16_t) * __SIZE_B;
	const long resLayoutSize = sizeof(int16_t) * __SIZE_RES;
	const unsigned long mem_size_a = aLayoutSize * qtd_a;
	const unsigned long mem_size_b = bLayoutSize * qtd_b;
	const unsigned long mem_size_pts = num_a * NConn * qtd_b * resLayoutSize;
	
	// allocate host memory
	int16_t *h_a = (int16_t *)(&a[0]);
	int16_t *h_b = (int16_t *)(&b[0]);
	int16_t *h_res = (int16_t *)malloc(mem_size_pts);

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	// setup execution parameters
	const int num_threads = 768; // 1024
	const int num_blocks = (qtd_b + num_threads -1) / num_threads;

	dim3 grid(num_blocks, num_a, NConn);
	dim3 threads(num_threads, 1, 1);

	// allocate device memory
	int16_t *d_a, *d_b, *d_pts;
	checkCudaErrors(cudaMalloc((void **)&d_a, mem_size_a));
	checkCudaErrors(cudaMalloc((void **)&d_b, mem_size_b));
	checkCudaErrors(cudaMalloc((void **)&d_pts, mem_size_pts));

	// copy host data to device
  	checkCudaErrors(cudaEventRecord(start));
	checkCudaErrors(cudaMemcpy(d_a, h_a, mem_size_a, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_b, h_b, mem_size_b, cudaMemcpyHostToDevice));

	for(int i = 0; i < qtd_a; i += num_a){
		int diff = qtd_a - i;
		if(diff < num_a){
			// std::cout << "qtd_a: " << diff << ", i: " << i << std::endl;
			createPts<<<grid, threads>>>(d_a, d_b, d_pts, diff, qtd_b, i);
		} else {
			// std::cout << "qtd_a: " << num_a << ", i: " << i << std::endl;
			createPts<<<grid, threads>>>(d_a, d_b, d_pts, num_a, qtd_b, i);
		}
	}

	// for(int i = 0; i < qtd_a; i += num_a){
	// 	int diff = qtd_a - i;
	// 	if(diff < num_a){
	// 		// std::cout << "qtd_a: " << diff << ", i: " << i << std::endl;
	// 		createPtsA<<<grid, threads>>>(d_a, d_pts, diff, qtd_b, i);
	// 		createPtsB<<<grid, threads>>>(d_a, d_b, d_pts, diff, qtd_b, i);
	// 	} else {
	// 		// std::cout << "qtd_a: " << num_a << ", i: " << i << std::endl;
	// 		createPtsA<<<grid, threads>>>(d_a, d_pts, num_a, qtd_b, i);
	// 		createPtsB<<<grid, threads>>>(d_a, d_b, d_pts, num_a, qtd_b, i);
	// 	}
	// }

	// createPts<<<grid, threads>>>(d_a, d_b, d_pts, num_a, qtd_b, 0);

	// cudaDeviceSynchronize();	
  	checkCudaErrors(cudaEventRecord(stop));
  	checkCudaErrors(cudaEventSynchronize(stop));

  	float msecTotal = 0.0f;
  	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	std::cout << "mem_size_a: " << mem_size_a << ", mem_size_b: " << mem_size_b << ", mem_size_pts: " << mem_size_pts << std::endl;
	std::cout << "mem_size_a (MB): " << ((float)mem_size_a)/1024.0/1024.0 << ", mem_size_b (MB): " << ((float)mem_size_b)/1024.0/1024.0 << ", mem_size_pts (MB): " << ((float)mem_size_pts)/1024.0/1024.0 << std::endl;

	std::cout << "a.size(): " << a.size() << ", b.size(): " << b.size() << std::endl;
	std::cout << "qtd_a: " << qtd_a << ", qtd_b: " << qtd_b << std::endl;

	std::cout << "num_threads: " << num_threads << ", num_blocks: " << num_blocks << std::endl;
	std::cout << "grid: " << grid.x << ", " << grid.y << ", " << grid.z << std::endl;
	std::cout << "threads: " << threads.x << ", " << threads.y << ", " << threads.z << std::endl;
	std::cout << "Time: " << msecTotal << std::endl;

	// check if kernel execution generated and error
	getLastCudaError("Kernel execution failed");

	// copy results from device to host
	checkCudaErrors(cudaMemcpy(h_res, d_pts, mem_size_pts, cudaMemcpyDeviceToHost));

	// cleanup device memory
	checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaFree(d_b));
	checkCudaErrors(cudaFree(d_pts));


	// std::cout << "A: " << std::endl;
	// for(int i = 0; i < qtd_a * __SIZE_A; i+=__SIZE_A){
	// 	for(int j = 0; j < __SIZE_A; j++){
	// 		std::cout << h_a[i + j] << ", ";
	// 	}
	// 	std::cout << std::endl;
	// }

	// std::cout << std::endl << "B: " << std::endl;
	// for(int i = 0; i < qtd_b * __SIZE_B; i+=__SIZE_B){
	// 	for(int j = 0; j < __SIZE_B; j++){
	// 		std::cout << h_b[i + j] << ", ";
	// 	}
	// 	std::cout << std::endl;
	// }

	// std::cout << std::endl << "Res: " << std::endl;
	// for(int i = 0; i < NConn; i++){
	// 	for(int j = 0; j < qtd_a; j++){
	// 		for(int k = 0; k < qtd_b; k++){
	// 			int baseIdx = ((i * qtd_a + j) * qtd_b + k) * __SIZE_RES;
	// 			std::cout << "NConn: " << i << ", a: " << j << ", b: " << k << ", idx: " << baseIdx << " :     ";
	// 			for(int l = 0; l < __SIZE_RES; l++, baseIdx++){
	// 				std::cout << h_res[baseIdx] << ", ";
	// 			}
	// 			std::cout << std::endl;
	// 		}
	// 	}
	// }

#ifdef OPENCV_ENABLED 
	int i = 0;
	const int max_i = (mem_size_pts / sizeof(int16_t))  - __SIZE_RES;
	std::vector<int16_t> PtsX(__SIZE_RES/2, 0);
	std::vector<int16_t> PtsY(__SIZE_RES/2, 0);
	while(i <= max_i){
		for(int j = 0; j < __SIZE_RES; j+=2){
			PtsX[j/2] = h_res[i + j];
			PtsY[j/2] = h_res[i + j + 1];
		}
		
		int c = CVHelper::showLayoutMove(PtsX, PtsY);
		// i +=__SIZE_RES * qtd_b * c;
		i += __SIZE_RES*c;
		if(i < 0)
			i = 0;
	}
#endif


	// cleanup host memory
	free(h_res);
}