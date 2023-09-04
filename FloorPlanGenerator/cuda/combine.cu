#include <cstdio>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <algorithm>
#include <filesystem>
#include <fstream>

#include "helper.cuh"
#include "combine.h"
#include "common.cuh"
#include "process.h"
#include "../lib/cvHelper.h"
#include "../lib/globals.h"


// #define _SIMPLE_DEBUG 
// #define _FULL_DEBUG

// Sorry, had to do it this way to make the reduce the cuda kernel registers usage

// #define check_overlap(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right) ((a_up >= b_up && a_up < b_down && a_left < b_right && a_left >= b_left) || (a_up >= b_up && a_up < b_down && a_right >= b_right && a_left <= b_left) || (a_up >= b_up && a_up < b_down && a_right <= b_right && a_right > b_left) || (a_down >= b_down && a_left < b_right && a_left >= b_left && a_up <= b_up) || (a_down >= b_down && a_up <= b_up && a_right <= b_right && a_right > b_left) || (a_left < b_right && a_left >= b_left && a_down > b_up && a_down <= b_down) || (a_right >= b_right && a_down > b_up && a_down <= b_down && a_left <= b_left) || (b_right >= a_right && b_up >= a_up && b_up < a_down && b_left <= a_left) || (b_right >= a_right && b_down > a_up && b_down <= a_down && b_left <= a_left) || (b_up >= a_up && b_up < a_down && b_left >= a_left && b_left < a_right) || (b_up >= a_up && b_up < a_down && b_right > a_left && b_right <= a_right) || (b_down >= a_down && b_left >= a_left && b_left < a_right && b_up <= a_up) || (b_down >= a_down && b_right > a_left && b_right <= a_right && b_up <= a_up) || (b_left >= a_left && b_left < a_right && b_down > a_up && b_down <= a_down) || (a_down > b_up && a_down <= b_down && a_right <= b_right && a_right > b_left) || (b_right > a_left && b_right <= a_right && b_down > a_up && b_down <= a_down))

// __device__
// uint8_t check_overlap(const int a_up, const int a_down, const int a_left, const int a_right, 
// 	const int b_up, const int b_down, const int b_left, const int b_right){
// 	if(((a_down > b_up && a_down <= b_down) ||
// 	(a_up  >= b_up && a_up < b_down)) &&
// 	((a_right > b_left && a_right <= b_right) ||
// 	(a_left  >= b_left && a_left  <  b_right) ||
// 	(a_left  <= b_left && a_right >= b_right))){
// 		return 0;
// 	}

// 	else if(((b_down > a_up && b_down <= a_down) ||
// 	(b_up >= a_up && b_up < a_down)) &&
// 	((b_right > a_left && b_right <= a_right) ||
// 	(b_left  >= a_left && b_left  <  a_right) ||
// 	(b_left  <= a_left && b_right >= a_right))){
// 		return 0;
// 	}

// 	else if(((a_right > b_left && a_right <= b_right) ||
// 	(a_left >= b_left && a_left < b_right)) &&
// 	((a_down > b_up && a_down <= b_down) ||
// 	(a_up  >= b_up && a_up   <  b_down) ||
// 	(a_up  <= b_up && a_down >= b_down))){
// 		return 0;
// 	}

// 	else if(((b_right > a_left && b_right <= a_right) ||
// 	(b_left >= a_left && b_left < a_right)) &&
// 	((b_down > a_up && b_down <= a_down) ||
// 	(b_up  >= a_up && b_up   <  a_down) ||
// 	(b_up  <= a_up && b_down >= a_down))){
// 		return 0;
// 	}

// 	return 1;
// }

// const int num_threads = __THREADS_PER_BLOCK
// const int num_blocks = (qtd_b + num_threads -1) / num_threads;
// dim3 grid(num_blocks, num_a, NConn);
// dim3 threads(num_threads, 1, 1);
__global__ 
void k_createPts(int16_t *d_a, int16_t *d_b, int *d_res, int *d_adj, const int qtd_a, const int qtd_b, const int a_offset) {
	// Block and thread indexes 	
	// Each blockIdx.x iterates over a fixed number (num_a) of A layouts (blockIdx.y), 
	// that iterates over Nconn connections (blockIdx.z). Each threadIdx.x represents
	// a Layout B design inside the blockIdx.x block 

	//K represents the connection (from 0 to 15, skipping 0, 5, 10 and 15)
	const int k = blockIdx.z + 1 + blockIdx.z/4; 
	int a_idx = blockIdx.y + a_offset; //layout A index
	int b_idx = blockIdx.x * blockDim.x + threadIdx.x; //layout B index
	const uint64_t res_idx = ((blockIdx.y * qtd_b * __N_CONN) + (b_idx * __N_CONN) + blockIdx.z) * __SIZE_RES;

	// Check bounds
	if(b_idx >= qtd_b || blockIdx.y >= qtd_a){
		return;
	}

	a_idx *= __SIZE_A_DISK;
	b_idx *= __SIZE_B_DISK;

	// Load A into shared memory
	__shared__ int16_t a[__SIZE_A_DISK];
	if(threadIdx.x < __SIZE_A_DISK){
		a[threadIdx.x] = d_a[a_idx + threadIdx.x];
	}
	
	__shared__ int req_adj[__SIZE_ADJ];
	if(threadIdx.x < __SIZE_ADJ){
		req_adj[threadIdx.x] = d_adj[threadIdx.x];
	}

  	__syncthreads();

	// Load B into local memory
	int16_t b[__SIZE_B_DISK];
	for(int i = 0; i < __SIZE_B_DISK; i++){
		b[i] = d_b[b_idx + i];
	}

	// Extract source and destination connections from k
	const int srcConn = k & 0b11;
	const int dstConn = (k >> 2) & 0b11;

	// Get X axis connection points from layout A and B
	int dst = 0;
	int src = 0;
	if(dstConn == 0 || dstConn == 2)
		dst = b[0];
	else 
		dst = b[2];

	if(srcConn == 0 || srcConn == 2)
		src = a[__SIZE_A_LAYOUT - 4];
	else 
		src = a[__SIZE_A_LAYOUT - 2];


	//Move layout B in the X axis by diffX points
	const int diffX = src - dst;
	for(int i = 0; i < __SIZE_B_LAYOUT; i+=2){
		b[i] += diffX;
	}

	// Get Y axis connection points from layout A and B
	if(dstConn == 0 || dstConn == 1)
		dst = b[1];
	else 
		dst = b[3];
		
	if(srcConn == 0 || srcConn == 1)
		src = a[__SIZE_A_LAYOUT - 3];
	else 
		src = a[__SIZE_A_LAYOUT - 1];

	//Move layout B in the Y axis by diffY points
	const int diffY = src - dst;
	for(int i = 1; i < __SIZE_B_LAYOUT; i+=2){
		b[i] += diffY;
	}

	// Find the bounding box of B
	int minH = 5000, maxH = -5000;
	int minW = 5000, maxW = -5000;
	for(int i = 0; i < __SIZE_B_LAYOUT; i+=4){
		if(b[i + __UP] < minH)
			minH = b[i + __UP];
		if(b[i + __DOWN] > maxH)
			maxH = b[i + __DOWN];
		if(b[i] < minW)
			minW = b[i];
		if(b[i + __RIGHT] > maxW)
			maxW = b[i + __RIGHT];
	}

	//left, up, right, down
	// Find the bounding box of A and check overlaping
	int connections[__N_A + __N_B];
	for(int i = 0; i < __N_A  + __N_B; i++){
		connections[i] = 1 << i;
	}

	for(int i = 0; i < __SIZE_A_LAYOUT; i+=4){
		const int a_left = a[i];
		const int a_up = a[i + __UP];
		const int a_down = a[i + __DOWN];
		const int a_right = a[i + __RIGHT];

		if(a_up < minH)
			minH = a_up;
		if(a_down > maxH)
			maxH = a_down;
		if(a_left < minW)
			minW = a_left;
		if(a_right > maxW)
			maxW = a_right;

		for(int j = 0; j < __SIZE_B_LAYOUT; j+=4){
			const int b_left = b[j];
			const int b_up = b[j + __UP];
			const int b_down = b[j + __DOWN];
			const int b_right = b[j + __RIGHT];

			if(!check_overlap(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right))
				return;
			
			if(check_adjacency(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right)){
				connections[i/4] |= 1 << (j/4) + __N_A;
				connections[(j/4) + __N_A] |= 1 << (i/4); 
			}
		}
	}

	for(int i = 0; i < __SIZE_A_LAYOUT; i+=4){
		const int a_left = a[i];
		const int a_up = a[i + __UP];
		const int a_down = a[i + __DOWN];
		const int a_right = a[i + __RIGHT];

		for(int j = 0; j < __SIZE_A_LAYOUT; j+=4){
			const int b_left = a[j];
			const int b_up = a[j + __UP];
			const int b_down = a[j + __DOWN];
			const int b_right = a[j + __RIGHT];

			if(check_adjacency(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right)){
				connections[i/4] |= 1 << (j/4);
				connections[j/4] |= 1 << (i/4); 
			}
		}
	}

	for(int i = 0; i < __SIZE_B_LAYOUT; i+=4){
		const int a_left = b[i];
		const int a_up = b[i + __UP];
		const int a_down = b[i + __DOWN];
		const int a_right = b[i + __RIGHT];

		for(int j = 0; j < __SIZE_B_LAYOUT; j+=4){
			const int b_left = b[j];
			const int b_up = b[j + __UP];
			const int b_down = b[j + __DOWN];
			const int b_right = b[j + __RIGHT];

			if(check_adjacency(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right)){
				connections[(i/4) + __N_A] |= 1 << ((j/4) + __N_A);
				connections[(j/4) + __N_A] |= 1 << ((i/4) + __N_A); 
			}
		}
	}

	const int a_perm_idx = a[__SIZE_A_LAYOUT];
	const int b_perm_idx = b[__SIZE_B_LAYOUT];

	int adj[__SIZE_ADJ_TYPES]; //Rid connections from the specific rId
	int adj_count[__SIZE_ADJ_TYPES]; //Idx of each room from the specific rId
	for(int i = 0; i < __SIZE_ADJ_TYPES; i++){
		adj[i] = 0;
		adj_count[i] = 0;
	}

	for(int i = 0; i < __N_A; i++){
		const int rplannyId = (a_perm_idx >> (i * __PERM_BITS_SIZE)) & __PERM_BITS;
		adj_count[rplannyId] |= 1 << i;
		adj[rplannyId] |= connections[i];
	}
	
	for(int i = 0; i < __N_B; i++){
		const int rplannyId = (b_perm_idx >> (i * __PERM_BITS_SIZE)) & __PERM_BITS;
		adj_count[rplannyId] |= 1 << (i + __N_A);
		adj[rplannyId] |= connections[i + __N_A];
	}

	for(int i = 0; i < __SIZE_ADJ_TYPES; i++){
		for(int j = 0; j < __SIZE_ADJ_TYPES; j++){
			const int req_adj_idx = i*__SIZE_ADJ_TYPES + j;
			if(req_adj[req_adj_idx] == REQ_ANY && !(adj[j] & adj_count[i]))
				return;

			if(req_adj[req_adj_idx] == REQ_ALL && (adj[j] & adj_count[i]) != adj_count[i])
				return;
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

	d_res[res_idx] = maxH - minH;
	d_res[res_idx + 1] = maxW - minW;
	d_res[res_idx + 2] = a_idx;
	d_res[res_idx + 3] = b_idx;
}

void gpuHandler::createPts(
		const std::vector<int16_t>& a, const std::vector<int16_t>& b,
    	std::vector<int> allReqAdj, std::string resultFolderPath, int id_a, int id_b) {
#ifdef _FULL_DEBUG
	const int qtd_a = 2;
	const int qtd_b = 12;
	const long num_a = 2;
	const int NConn = __N_CONN;
#else
	const int NConn = __N_CONN;  	// always 12. Qtd of valid connectction between two rooms
	const int qtd_a = a.size() / __SIZE_A_DISK;
	const int qtd_b = b.size() / __SIZE_B_DISK;
	const int num_a = qtd_a > 200 ? 200 : qtd_a;	//
#endif

	findCudaDevice();	
	const long qtd_res = num_a * NConn * qtd_b;

	const long aLayoutSize = sizeof(int16_t) * __SIZE_A_DISK;
	const long bLayoutSize = sizeof(int16_t) * __SIZE_B_DISK;
	const long resLayoutSize = sizeof(int) * __SIZE_RES;
	
	const unsigned long mem_size_a = aLayoutSize * qtd_a;
	const unsigned long mem_size_b = bLayoutSize * qtd_b;
	const unsigned long mem_size_res = resLayoutSize * qtd_res;
	const unsigned long mem_size_adj = sizeof(int) * __SIZE_ADJ;

	// allocate host memory (CPU)
	int *h_res = nullptr;
	int *h_adj = (int *)(&allReqAdj[0]);
	int16_t *h_a = (int16_t *)(&a[0]);
	int16_t *h_b = (int16_t *)(&b[0]);
	cudaMallocHost((void**)&h_res, mem_size_res);

#ifdef _SIMPLE_DEBUG
	// Allocate CUDA events used for timing
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
#endif

	// setup execution parameters
	const int num_threads = qtd_b > __THREADS_PER_BLOCK ? __THREADS_PER_BLOCK : qtd_b; 
	const int num_blocks = (qtd_b + num_threads -1) / num_threads;

	dim3 grid(num_blocks, num_a, NConn);
	dim3 threads(num_threads, 1, 1);

	// allocate device memory
	int *d_adj, *d_res;
	int16_t *d_a, *d_b;
	checkCudaErrors(cudaMalloc((void **)&d_a, mem_size_a));
	checkCudaErrors(cudaMalloc((void **)&d_b, mem_size_b));
	checkCudaErrors(cudaMalloc((void **)&d_res, mem_size_res));
	checkCudaErrors(cudaMalloc((void **)&d_adj, mem_size_adj));
	checkCudaErrors(cudaMemset(d_res, 0, mem_size_res));

	// copy host data to device
#ifdef _SIMPLE_DEBUG
  	checkCudaErrors(cudaEventRecord(start));
#endif

	checkCudaErrors(cudaMemcpy(d_a, h_a, mem_size_a, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_b, h_b, mem_size_b, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_adj, h_adj, mem_size_adj, cudaMemcpyHostToDevice));

	const int max_layout_size = 200;
	std::vector<int> result;
	std::vector<int> h_begin(max_layout_size, 0);
	std::vector<int> index_table(max_layout_size * max_layout_size, 0); //relative

	// k_createPts<<<grid, threads>>>(d_a, d_b, d_res, d_adj, num_a, qtd_b, 0);
	// checkCudaErrors(cudaMemcpy(h_res, d_res, mem_size_res, cudaMemcpyDeviceToHost));
	// CudaProcess::processResult(result, h_res, qtd_res, h_begin, index_table, max_layout_size);
	
	for(int i = 0; i < qtd_a; i += num_a){
		int diff = qtd_a - i; 
		#ifdef _SIMPLE_DEBUG
			std::cout << (float)i / (float)qtd_a <<  " %" << std::endl;
		#endif
		
		if(diff < num_a){
			k_createPts<<<grid, threads>>>(d_a, d_b, d_res, d_adj, diff, qtd_b, i);
			checkCudaErrors(cudaMemcpy(h_res, d_res, mem_size_res, cudaMemcpyDeviceToHost));
			CudaProcess::processResult(result, h_res, qtd_res, h_begin, index_table, max_layout_size);
		} else {
			k_createPts<<<grid, threads>>>(d_a, d_b, d_res, d_adj, num_a, qtd_b, i);
			checkCudaErrors(cudaMemcpy(h_res, d_res, mem_size_res, cudaMemcpyDeviceToHost));
			CudaProcess::processResult(result, h_res, qtd_res, h_begin, index_table, max_layout_size);
		}
	}

#ifdef _SIMPLE_DEBUG
  	checkCudaErrors(cudaEventRecord(stop));
  	checkCudaErrors(cudaEventSynchronize(stop));

  	float msecTotal = 0.0f;
  	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
#else
	cudaDeviceSynchronize();
#endif

#ifdef _SIMPLE_DEBUG
std::cout << std::endl;
	std::cout << "a.size(): " << a.size() << ", b.size(): " << b.size() << std::endl;
	std::cout << "qtd_a: " << qtd_a << ", qtd_b: " << qtd_b  << ", a*b: " << qtd_a * qtd_b << std::endl;
	std::cout << "num_threads: " << num_threads << ", num_blocks: " << num_blocks << std::endl;
	std::cout << "grid: " << grid.x << ", " << grid.y << ", " << grid.z << std::endl;
	std::cout << "threads: " << threads.x << ", " << threads.y << ", " << threads.z << std::endl;
	std::cout << "mem_size_a: " << mem_size_a << ", mem_size_b: " << mem_size_b << ", mem_size_res: " << mem_size_res << ", mem_size_adj: " << mem_size_adj << std::endl;
	std::cout << "mem_size_a (MB): " << ((float)mem_size_a)/1024.0/1024.0 << ", mem_size_b (MB): " << ((float)mem_size_b)/1024.0/1024.0 << ", mem_size_res (MB): " << ((float)mem_size_res)/1024.0/1024.0 << std::endl;
	std::cout << "Time: " << msecTotal << std::endl;
#endif


    // std::string result_data_path = resultFolderPath + "/" + std::to_string(id_a | (id_b << 16)) + ".dat";
    // std::ofstream outputFile(result_data_path, std::ios::out | std::ios::binary);
    // outputFile.write(reinterpret_cast<const char*>(result.data()), result.size() * sizeof(int16_t));
    // outputFile.close();

	// check if kernel execution generated and error
	getLastCudaError("Kernel execution failed");

	// cleanup device memory
	checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaFree(d_b));
	checkCudaErrors(cudaFree(d_res));

// #ifdef _SIMPLE_DEBUG
// 	for(int i = 0; i < num_a * NConn * qtd_b; i++){
// 		int memAddr = i * __SIZE_RES;
// 		if(h_res[memAddr] == 0)
// 			continue;

// 		std::cout << "i: " << i << ", memAddr: " << memAddr << std::endl;
// 		for(int j = 0; j < __SIZE_RES; j++){
// 			std::cout << h_res[memAddr + j] << ", ";
// 		}std::cout << std::endl;

// 		getchar();
// 	}
// #endif

	// cleanup host memory
	checkCudaErrors(cudaFreeHost(h_res));
}