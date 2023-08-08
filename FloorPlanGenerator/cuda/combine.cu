#include <cstdio>
#include <iostream>
#include <vector>
#include <stdint.h>

#include "combine.h"
#include "helper.h"
#include "../lib/cvHelper.h"
#include "../lib/globals.h"

#define __N_A 3
#define __N_B 3
#define __N_CONN 12 // always 12

#define __CONN_CHECK 63 //1 << 0 | 1 << 1 | .... | 1 << (_N_A + N_B - 1)
#define __CONN_CHECK_IDX 5 // _N_A + N_B - 1

#define __SIZE_A 12		// n_a * 4
#define __SIZE_B 12		// n_b * 4
#define __SIZE_A_DISK 13 // __SIZE_B + perm iter value
#define __SIZE_B_DISK 13 // __SIZE_B + perm iter value
#define __SIZE_RES 2

#define __LEFT 0
#define __UP 1
#define __RIGHT 2
#define __DOWN 3

#define __THREADS_PER_BLOCK 768 // 192, 288, 384, 480, 576, 672, 768, 862, 

// #define _SIMPLE_DEBUG 
// #define _FULL_DEBUG

__device__
uint8_t check_overlap(const int a_up, const int a_down, const int a_left, const int a_right, 
					const int b_up, const int b_down, const int b_left, const int b_right){
	if(((a_down > b_up && a_down <= b_down) ||
		(a_up  >= b_up && a_up < b_down)) &&
		((a_right > b_left && a_right <= b_right) ||
		(a_left  >= b_left && a_left  <  b_right) ||
		(a_left  <= b_left && a_right >= b_right))){
			return 0;
	}
	
	else if(((b_down > a_up && b_down <= a_down) ||
		(b_up >= a_up && b_up < a_down)) &&
		((b_right > a_left && b_right <= a_right) ||
		(b_left  >= a_left && b_left  <  a_right) ||
		(b_left  <= a_left && b_right >= a_right))){
			return 0;
	}

	else if(((a_right > b_left && a_right <= b_right) ||
		(a_left >= b_left && a_left < b_right)) &&
		((a_down > b_up && a_down <= b_down) ||
		(a_up  >= b_up && a_up   <  b_down) ||
		(a_up  <= b_up && a_down >= b_down))){
			return 0;
	}
	
	else if(((b_right > a_left && b_right <= a_right) ||
		(b_left >= a_left && b_left < a_right)) &&
		((b_down > a_up && b_down <= a_down) ||
		(b_up  >= a_up && b_up   <  a_down) ||
		(b_up  <= a_up && b_down >= a_down))){
			return 0;
	}

	return 1;
}

__device__
uint8_t check_adjacency(const int a_up, const int a_down, const int a_left, const int a_right, 
					const int b_up, const int b_down, const int b_left, const int b_right){    
	if((a_down == b_up || a_up == b_down) &&
        ((a_right > b_left && a_right <= b_right) ||
        (a_left < b_right && a_left >= b_left) ||
        (a_left <= b_left && a_right >= b_right)))
            return 1;   

    if((a_left == b_right || a_right == b_left) &&
        ((a_down > b_up && a_down <= b_down) ||
        (a_up < b_down && a_up >= b_up) ||
        (a_up <= b_up && a_down >= b_down)))
            return 1; 

    return 0;
}

// const int num_threads = __THREADS_PER_BLOCK
// const int num_blocks = (qtd_b + num_threads -1) / num_threads;
// dim3 grid(num_blocks, num_a, NConn);
// dim3 threads(num_threads, 1, 1);
__global__ 
void k_createPts(int16_t *d_a, int16_t *d_b, int16_t *d_res, const int qtd_a, const int qtd_b, const int a_offset) {
	// Block and thread indexes
	// Each blockIdx.x iterates over a fixed number (num_a) of A layouts (blockIdx.y), 
	// that iterates over Nconn connections (blockIdx.z). Each threadIdx.x represents
	// a Layout B design inside the blockIdx.x block 

	//K represents the connection (from 0 to 15, skipping 0, 5, 10 and 15)
	const int k = blockIdx.z + 1 + blockIdx.z/4; 
	const int a_idx = (blockIdx.y + a_offset) * __SIZE_A_DISK; //layout A index
	int b_idx = blockIdx.x * blockDim.x + threadIdx.x; //layout B index (without * __SIZE_B)
	const int res_idx = ((blockIdx.z * qtd_a + blockIdx.y) * qtd_b +  b_idx) * __SIZE_RES;

	// Check bounds
	if(b_idx >= qtd_b || blockIdx.y >= qtd_a)
		return;

	// printf("x: %3d, y: %3d, z: %2d, tx: %3d, k: %2d, a: %6d, b: %6d, res: %7d\n",blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, k, a_idx, b_idx, res_idx);
	// return;
	

// #ifdef _FULL_DEBUG
// 	printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, k: %d, a_idx: %d, b_idx: %d, res_idx: %d\n",blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, k, a_idx, b_idx, res_idx);
// 	if(threadIdx.x == 1 && k == 1){
// 		// printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, k, a_idx, b_idx, res_idx);
// 	}
// 	return;
// #endif

	// Load A into shared memory
	__shared__ int16_t a[__SIZE_A_DISK];
	if(threadIdx.x < __SIZE_A_DISK){
		a[threadIdx.x] = d_a[a_idx + threadIdx.x];
	}

  	__syncthreads();

	b_idx *= __SIZE_B_DISK;
	// Load B into local memory
	int16_t b[__SIZE_B_DISK];
	for(int i = 0; i < __SIZE_B_DISK; i++){
		b[i] = d_b[b_idx + i];
	}

	
// #ifdef _FULL_DEBUG
// 	if(threadIdx.x == 1 && k == 1){
// 		printf("A: ");
// 		for(int i = 0; i < __SIZE_A; i++){
// 			printf("%d, ", a[i]);
// 		}

// 		printf("\nB: ");
// 		for(int i = 0; i < __SIZE_B; i++){
// 			printf("%d, ", b[i]);
// 		}
// 		printf("\n");
// 	}
// #endif
	
	// Extract source and destination connections from k
	const int srcConn = k & 0b11;
	const int dstConn = (k >> 2) & 0b11;
	
// #ifdef _FULL_DEBUG
// 	if(threadIdx.x == 1 && k == 1)
// 		printf("srcConn: %d, dstConn: %d\n", srcConn, dstConn);
// #endif

	// Get X axis connection points from layout A and B
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


	//Move layout B in the X axis by diffX points
	const int diffX = src - dst;
	for(int i = 0; i < __SIZE_B; i+=2){
		b[i] += diffX;
	}
	// d_res[a_idx + b_idx + k - a_idx - b_idx - k] = 1;
	// return;

// #ifdef _FULL_DEBUG
// 	if(threadIdx.x == 1 && k == 1)
// 		printf("dst: %d, src: %d, diffX: %d\n", dst, src, diffX);
// #endif

	// Get Y axis connection points from layout A and B
	if(dstConn == 0 || dstConn == 1)
		dst = b[1];
	else 
		dst = b[3];
		
	if(srcConn == 0 || srcConn == 1)
		src = a[__SIZE_A - 3];
	else 
		src = a[__SIZE_A - 1];

	//Move layout B in the Y axis by diffY points
	const int diffY = src - dst;
	for(int i = 1; i < __SIZE_B; i+=2){
		b[i] += diffY;
	}

// #ifdef _SIMPLE_DEBUG
// 	if(res_idx < 12 ){
// 		printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, " \
// 		"k: %d, a_idx: %d, b_idx: %d, res_idx: %d\n, " \
// 		"(%hd, %hd), (%hd, %hd)\n, " \
// 		"(%hd, %hd), (%hd, %hd)\n, " \
// 		"(%hd, %hd), (%hd, %hd)\n, " \
// 		"(%hd, %hd), (%hd, %hd)\n, " \
// 		"(%hd, %hd), (%hd, %hd)\n, " \
// 		"(%hd, %hd), (%hd, %hd)\n, " \
// 		"\n\n\n",
// 		blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, 
// 		k, a_idx, b_idx, res_idx,
// 		a[0], a[1], a[2], a[3], 
// 		a[4], a[5], a[6], a[7], 
// 		a[8], a[9], a[10], a[11],
// 		b[0], b[1], b[2], b[3], 
// 		b[4], b[5], b[6], b[7], 
// 		b[8], b[9], b[10], b[11]);
// 	}
// #endif

// #ifdef _FULL_DEBUG
// 	if(threadIdx.x == 1 && k == 1)
// 		printf("dst: %d, src: %d, diffY: %d\n", dst, src, diffY);
// #endif

	// Find the bounding box of B
	int16_t minH = 5000, maxH = -5000;
	int16_t minW = 5000, maxW = -5000;
	for(int i = 0; i < __SIZE_B; i+=4){
		if(b[i + __UP] < minH)
			maxH = b[i + __UP];
		if(b[i + __DOWN] > maxH)
			maxH = b[i + __DOWN];
		if(b[i] < minW)
			minW = b[i];
		if(b[i + __RIGHT] > maxW)
			maxW = b[i + __RIGHT];
	}

	//left, up, right, down
	// Find the bounding box of A and check overlaping
	int notOverlap = 1;
	int connections[__N_A + __N_B];
	for(int i = 0; i < __N_A  + __N_B; i++){
		connections[i] = 1 << i;
	}

	for(int i = 0; i < __SIZE_A && notOverlap; i+=4){
		const int16_t a_left = a[i];
		const int16_t a_up = a[i + __UP];
		const int16_t a_down = a[i + __DOWN];
		const int16_t a_right = a[i + __RIGHT];

		if(a_up < minH)
			minH = a_up;
		if(a_down > maxH)
			maxH = a_down;
		if(a_left < minW)
			minW = a_left;
		if(a_right > maxW)
			maxW = a_right;

		for(int j = 0; j < __SIZE_B && notOverlap; j+=4){
			const int16_t b_left = b[j];
			const int16_t b_up = b[j + __UP];
			const int16_t b_down = b[j + __DOWN];
			const int16_t b_right = b[j + __RIGHT];

			if(b_up < minH)
				minH = b_up;
			if(b_down > maxH)
				maxH = b_down;
			if(b_left < minW)
				minW = b_left;
			if(b_right > maxW)
				maxW = b_right;

			notOverlap = check_overlap(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right);
			
			if(check_adjacency(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right)){
				connections[i/4] |= 1 << (j/4) + __N_A;
				connections[(j/4) + __N_A] |= 1 << (i/4); 
			}
		}
	}

	if(!notOverlap)
		return;

	for(int i = 0; i < __SIZE_A; i+=4){
		const int16_t a_left = a[i];
		const int16_t a_up = a[i + __UP];
		const int16_t a_down = a[i + __DOWN];
		const int16_t a_right = a[i + __RIGHT];

		for(int j = 0; j < __SIZE_A; j+=4){
			const int16_t b_left = a[j];
			const int16_t b_up = a[j + __UP];
			const int16_t b_down = a[j + __DOWN];
			const int16_t b_right = a[j + __RIGHT];

			if(check_adjacency(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right)){
				connections[i/4] |= 1 << (j/4);
				connections[j/4] |= 1 << (i/4); 
			}
		}
	}

	for(int i = 0; i < __SIZE_B; i+=4){
		const int16_t a_left = b[i];
		const int16_t a_up = b[i + __UP];
		const int16_t a_down = b[i + __DOWN];
		const int16_t a_right = b[i + __RIGHT];

		for(int j = 0; j < __SIZE_B; j+=4){
			const int16_t b_left = b[j];
			const int16_t b_up = b[j + __UP];
			const int16_t b_down = b[j + __DOWN];
			const int16_t b_right = b[j + __RIGHT];

			if(check_adjacency(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right)){
				connections[(i/4) + __N_A] |= 1 << ((j/4) + __N_A);
				connections[(j/4) + __N_A] |= 1 << ((i/4) + __N_A); 
			}
		}
	}

	for(int i = 0; i < __N_A + __N_B; i++){
		const int conns = connections[i];
		for(int j = i + 1; j < __N_A + __N_B; j++){
			if(connections[j] & 1 << i)
				connections[j] |= conns;
		}
	}

	if(connections[__CONN_CHECK_IDX] != __CONN_CHECK)
		return;

	// #ifdef _SIMPLE_DEBUG
	// 	if(res_idx < 300){
	// 		// printf("2 - %d,\t %d\t %d\t %d\t %d\t %d\t %d\t\n", res_idx/2, connections[0], connections[1], connections[2], connections[3], connections[4], connections[5]);
	// 		printf("%d %d\n", connections[5] == __CONN_CHECK, res_idx/2);
	// 	}
	// #endif

	d_res[res_idx] = maxH - minH;
	d_res[res_idx + 1] = maxW - minW;

	// #ifdef _FULL_DEBUG
	// 	if(threadIdx.x == 1 && k == 1){
	// 		printf("%d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\n", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
	// 		printf("%d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\n", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11]);
	// 		printf("%d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\n", b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11]);
	// 		printf("%d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\n", d_res[res_idx + 0], d_res[res_idx + 1], d_res[res_idx + 2], d_res[res_idx + 3], d_res[res_idx + 4], d_res[res_idx + 5], d_res[res_idx + 6], d_res[res_idx + 7], d_res[res_idx + 8], d_res[res_idx + 9], d_res[res_idx + 10], d_res[res_idx + 11]);
	// 		printf("%d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\n", d_res[res_idx + 12], d_res[res_idx + 13], d_res[res_idx + 14], d_res[res_idx + 15], d_res[res_idx + 16], d_res[res_idx + 17], d_res[res_idx + 18], d_res[res_idx + 19], d_res[res_idx + 20], d_res[res_idx + 21], d_res[res_idx + 22], d_res[res_idx + 23]);
	// 	}
	// #endif
}


void gpuHandler::createPts(
		const std::vector<int16_t>& a, const std::vector<int16_t>& b,
		std::vector<RoomConfig> setupsA, std::vector<RoomConfig> setupsB,
    	std::vector<int> allReq) {
#ifdef _FULL_DEBUG
	const int qtd_a = 2;
	const int qtd_b = 12;
	const long num_a = 2;
	const int NConn = __N_CONN;
#else
	const int NConn = __N_CONN;  	// always 12. Qtd of valid connectction between two rooms
	const long num_a = 200;	//
	const int qtd_a = a.size() / __SIZE_A_DISK;
	const int qtd_b = b.size() / __SIZE_B_DISK;
#endif

	findCudaDevice();	

	const long aLayoutSize = sizeof(int16_t) * __SIZE_A_DISK;
	const long bLayoutSize = sizeof(int16_t) * __SIZE_B_DISK;
	const long resLayoutSize = sizeof(int16_t) * __SIZE_RES;
	const unsigned long mem_size_a = aLayoutSize * qtd_a;
	const unsigned long mem_size_b = bLayoutSize * qtd_b;
	const unsigned long mem_size_res = num_a * NConn * qtd_b * resLayoutSize;

	// allocate host memory (CPU)
	int16_t *h_a = (int16_t *)(&a[0]);
	int16_t *h_b = (int16_t *)(&b[0]);
	int16_t *h_res = nullptr;

	cudaMallocHost((void**)&h_res, mem_size_res);

#ifdef _SIMPLE_DEBUG
	// Allocate CUDA events used for timing
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
#endif

	// setup execution parameters
	const int num_threads = __THREADS_PER_BLOCK; 
	const int num_blocks = (qtd_b + num_threads -1) / num_threads;

	dim3 grid(num_blocks, num_a, NConn);
	dim3 threads(num_threads, 1, 1);

	// allocate device memory
	int16_t *d_a, *d_b, *d_res;
	checkCudaErrors(cudaMalloc((void **)&d_a, mem_size_a));
	checkCudaErrors(cudaMalloc((void **)&d_b, mem_size_b));
	checkCudaErrors(cudaMalloc((void **)&d_res, mem_size_res));
	checkCudaErrors(cudaMemset(d_res, 0, mem_size_res));

	// copy host data to device
#ifdef _SIMPLE_DEBUG
  	checkCudaErrors(cudaEventRecord(start));
#endif

	checkCudaErrors(cudaMemcpy(d_a, h_a, mem_size_a, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_b, h_b, mem_size_b, cudaMemcpyHostToDevice));

	k_createPts<<<grid, threads>>>(d_a, d_b, d_res, num_a, qtd_b, 0);
	// for(int i = 0; i < qtd_a; i += num_a){
	// 	int diff = qtd_a - i;
	// 	if(diff < num_a){
	// 		k_createPts<<<grid, threads>>>(d_a, d_b, d_res, diff, qtd_b, i);
	// 		// cudaMemcpy(h_res, d_res, mem_size_res, cudaMemcpyDeviceToHost);
	// 	} else {
	// 		k_createPts<<<grid, threads>>>(d_a, d_b, d_res, num_a, qtd_b, i);
	// 		// cudaMemcpy(h_res, d_res, mem_size_res, cudaMemcpyDeviceToHost);
	// 	}
	// }

#ifdef _SIMPLE_DEBUG
  	checkCudaErrors(cudaEventRecord(stop));
  	checkCudaErrors(cudaEventSynchronize(stop));

  	float msecTotal = 0.0f;
  	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
#else
	cudaDeviceSynchronize();
#endif

#ifdef _SIMPLE_DEBUG
	std::cout << "a.size(): " << a.size() << ", b.size(): " << b.size() << std::endl;
	std::cout << "qtd_a: " << qtd_a << ", qtd_b: " << qtd_b  << ", a*b: " << qtd_a * qtd_b << std::endl;
	std::cout << "num_threads: " << num_threads << ", num_blocks: " << num_blocks << std::endl;
	std::cout << "grid: " << grid.x << ", " << grid.y << ", " << grid.z << std::endl;
	std::cout << "threads: " << threads.x << ", " << threads.y << ", " << threads.z << std::endl;
	std::cout << "mem_size_a: " << mem_size_a << ", mem_size_b: " << mem_size_b << ", mem_size_res: " << mem_size_res << std::endl;
	std::cout << "mem_size_a (MB): " << ((float)mem_size_a)/1024.0/1024.0 << ", mem_size_b (MB): " << ((float)mem_size_b)/1024.0/1024.0 << ", mem_size_res (MB): " << ((float)mem_size_res)/1024.0/1024.0 << std::endl;
	std::cout << "Time: " << msecTotal << std::endl;
#endif

	// check if kernel execution generated and error
	getLastCudaError("Kernel execution failed");

	// copy results from device to host
	checkCudaErrors(cudaMemcpy(h_res, d_res, mem_size_res, cudaMemcpyDeviceToHost));

	// cleanup device memory
	checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaFree(d_b));
	checkCudaErrors(cudaFree(d_res));

#ifdef _SIMPLE_DEBUG
	for(int i = 0; i < num_a * NConn * qtd_b; i++){
		int memAddr = i * __SIZE_RES;
		std::cout << "i: " << i << ", memAddr: " << memAddr << std::endl;
		// std::cout << "i: " << i << ", res: " << h_res[memAddr] << 
		// ", maxH: " << h_res[memAddr + 1] << ", maxW: " << h_res[memAddr + 2] << 
		// ", minH: " << h_res[memAddr + 3] << ", minW: " << h_res[memAddr + 4] << std::endl;

		std::cout << "i: " << i << "\t";
		for(int j = 0; j < __SIZE_RES; j++){
			std::cout << h_res[memAddr + j] << ", ";
		}
		std::cout << std::endl;

		getchar();
	}
#endif

	// cleanup host memory
	checkCudaErrors(cudaFreeHost(h_res));
}