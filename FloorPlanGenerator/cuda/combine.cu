#include <cstdio>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <algorithm>
#include <filesystem>
#include <fstream>

#include "helper.cuh"
#include "combine.cuh"
#include "common.cuh"
#include "process.h"
#include "../lib/cvHelper.h"
#include "../lib/globals.h"



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


__global__ 
void k_createPts(int16_t *d_a, int16_t *d_b, int *d_res, int *d_adj, int *d_conn, const int qtd_a, const int qtd_b, const int a_offset) {
	// Block and thread indexes 	
	// Each blockIdx.x iterates over a fixed number (num_a) of A layouts (blockIdx.y), 
	// that iterates over Nconn connections (blockIdx.z). Each threadIdx.x represents
	// a Layout B design inside the blockIdx.x block 

	//K represents the connection (from 0 to 15, skipping 0, 5, 10 and 15)
	// const int kidx = blockIdx.z; 
	int a_idx = blockIdx.y + a_offset; //layout A index
	int b_idx = blockIdx.x * blockDim.x + threadIdx.x; //layout B index
	const uint64_t res_idx = ((blockIdx.y * qtd_b * gridDim.z) + (b_idx * gridDim.z) + blockIdx.z) * __SIZE_RES;

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

	__shared__ int k;
	if(threadIdx.x == 0){
		k = d_conn[blockIdx.z];
	}

  	__syncthreads();

	// if(a_idx != 0 || b_idx != 0)
	// 	return;

	// if(blockIdx.z > 0)
	// 	return;

	// Load B into local memory
	int16_t b[__SIZE_B_DISK];
	for(int i = 0; i < __SIZE_B_DISK; i++){
		b[i] = d_b[b_idx + i];
	}

	// Extract source and destination connections from k
	int srcConn = (k >> __COMBINE_CONN_SRC_X_SHIFT) & __COMBINE_CONN_BITS;
	int dstConn = (k >> __COMBINE_CONN_DST_X_SHIFT) & __COMBINE_CONN_BITS;



	// printf("1 - %ld - pts:\n(%d, %d), (%d, %d)\n(%d, %d), (%d, %d)\n(%d, %d), (%d, %d)\n(%d, %d), (%d, %d)\n(%d, %d), (%d, %d)\n(%d, %d), (%d, %d)\n\n", 
	// res_idx,
	// a[0], a[1], a[2], a[3], 
	// a[4], a[5], a[6], a[7], 
	// a[8], a[9], a[10], a[11],
	// b[0], b[1], b[2], b[3], 
	// b[4], b[5], b[6], b[7], 
	// b[8], b[9], b[10], b[11]);

	int src = a[srcConn];
	int dst = b[dstConn];
	const int diffX = src - dst;

	srcConn = (k >> __COMBINE_CONN_SRC_Y_SHIFT) & __COMBINE_CONN_BITS;
	dstConn = (k >> __COMBINE_CONN_DST_Y_SHIFT) & __COMBINE_CONN_BITS;
	src = a[srcConn];
	dst = b[dstConn];
	const int diffY = src - dst;

	// printf("\n\nbx: %d, by: %d, bz: %d, tx: %d, ty: %d, tz: %d\nres: %ld, a_idx: %d, b_idx: %d\nk: %d, src X: %d (%d), src Y: %d (%d), dst X: %d (%d), dst Y: %d (%d)\ndiffX: %d, diffY: %d, \n\n",
	// 		blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,
	// 		res_idx, blockIdx.y + a_offset, blockIdx.x * blockDim.x + threadIdx.x,
	// 		k,
	// 		(k >> __COMBINE_CONN_SRC_X_SHIFT) & __COMBINE_CONN_BITS, a[(k >> __COMBINE_CONN_SRC_X_SHIFT) & __COMBINE_CONN_BITS],
	// 		(k >> __COMBINE_CONN_SRC_Y_SHIFT) & __COMBINE_CONN_BITS, a[(k >> __COMBINE_CONN_SRC_Y_SHIFT) & __COMBINE_CONN_BITS],
	// 		(k >> __COMBINE_CONN_DST_X_SHIFT) & __COMBINE_CONN_BITS, b[(k >> __COMBINE_CONN_DST_X_SHIFT) & __COMBINE_CONN_BITS],
	// 		(k >> __COMBINE_CONN_DST_Y_SHIFT) & __COMBINE_CONN_BITS, b[(k >> __COMBINE_CONN_DST_Y_SHIFT) & __COMBINE_CONN_BITS],
	// 		diffX, diffY);

	//Move layout B in the X and Y axis by diffX and diffY points
	for(int i = 0; i < __SIZE_B_LAYOUT; i+=2){
		b[i] += diffX;
		b[i + 1] += diffY;
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
	int connections[__COMBINE_N_A + __COMBINE_N_B];
	for(int i = 0; i < __COMBINE_N_A  + __COMBINE_N_B; i++){
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

			// if(!check_overlap(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right))
			// 	return;
			
			if(check_adjacency(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right)){
				connections[i/4] |= 1 << (j/4) + __COMBINE_N_A;
				connections[(j/4) + __COMBINE_N_A] |= 1 << (i/4); 
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
				connections[(i/4) + __COMBINE_N_A] |= 1 << ((j/4) + __COMBINE_N_A);
				connections[(j/4) + __COMBINE_N_A] |= 1 << ((i/4) + __COMBINE_N_A); 
			}
		}
	}

	const int a_rid_idx = a[__SIZE_A_LAYOUT];
	const int b_rid_idx = b[__SIZE_B_LAYOUT];

	int adj[__SIZE_ADJ_TYPES]; //Rid connections from the specific rId
	int adj_count[__SIZE_ADJ_TYPES]; //Idx of each room from the specific rId
	for(int i = 0; i < __SIZE_ADJ_TYPES; i++){
		adj[i] = 0;
		adj_count[i] = 0;
	}

	for(int i = 0; i < __COMBINE_N_A; i++){
		const int rplannyId = (a_rid_idx >> (i * __RID_BITS_SIZE)) & __RID_BITS;
		adj_count[rplannyId] |= 1 << i;
		adj[rplannyId] |= connections[i];
	}
	
	for(int i = 0; i < __COMBINE_N_B; i++){
		const int rplannyId = (b_rid_idx >> (i * __RID_BITS_SIZE)) & __RID_BITS;
		adj_count[rplannyId] |= 1 << (i + __COMBINE_N_A);
		adj[rplannyId] |= connections[i + __COMBINE_N_A];
	}

	for(int i = 0; i < __SIZE_ADJ_TYPES; i++){
		for(int j = 0; j < __SIZE_ADJ_TYPES; j++){
			const int req_adj_idx = i*__SIZE_ADJ_TYPES + j;
			// if(req_adj[req_adj_idx] == REQ_ANY && !(adj[j] & adj_count[i]))
			// 	return;

			// if(req_adj[req_adj_idx] == REQ_ALL && (adj[j] & adj_count[i]) != adj_count[i])
			// 	return;
		}
	}

	for(int i = 0; i < __COMBINE_N_A + __COMBINE_N_B; i++){
		const int conns = connections[i];
		for(int j = i + 1; j < __COMBINE_N_A + __COMBINE_N_B; j++){
			if(connections[j] & 1 << i)
				connections[j] |= conns;
		}
	}

	// if(connections[__CONN_CHECK_IDX] != __CONN_CHECK)
	// 	return;

	// if(res_idx == 1660128){
	// 	printf("\n\nbx: %d, by: %d, bz: %d, tx: %d, ty: %d, tz: %d\n\n",
	// 			blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);


	// printf("2 - %ld - pts:\n(%d, %d), (%d, %d)\n(%d, %d), (%d, %d)\n(%d, %d), (%d, %d)\n(%d, %d), (%d, %d)\n(%d, %d), (%d, %d)\n(%d, %d), (%d, %d)\n\n", 
	// res_idx,
	// a[0], a[1], a[2], a[3], 
	// a[4], a[5], a[6], a[7], 
	// a[8], a[9], a[10], a[11],
	// b[0], b[1], b[2], b[3], 
	// b[4], b[5], b[6], b[7], 
	// b[8], b[9], b[10], b[11]);
	// }

	d_res[res_idx] = a[0];
	d_res[res_idx + 1] = a[1];
	d_res[res_idx + 2] = a[2];
	d_res[res_idx + 3] = a[3];
	d_res[res_idx + 4] = a[4];
	d_res[res_idx + 5] = a[5];
	d_res[res_idx + 6] = a[6];
	d_res[res_idx + 7] = a[7];
	d_res[res_idx + 8] = a[8];
	d_res[res_idx + 9] = a[9];
	d_res[res_idx + 10] = a[10];
	d_res[res_idx + 11] = a[11];

	d_res[res_idx + 12] = b[0];
	d_res[res_idx + 13] = b[1];
	d_res[res_idx + 14] = b[2];
	d_res[res_idx + 15] = b[3];
	d_res[res_idx + 16] = b[4];
	d_res[res_idx + 17] = b[5];
	d_res[res_idx + 18] = b[6];
	d_res[res_idx + 19] = b[7];
	d_res[res_idx + 20] = b[8];
	d_res[res_idx + 21] = b[9];
	d_res[res_idx + 22] = b[10];
	d_res[res_idx + 23] = b[11];
	// d_res[res_idx] = maxH - minH;
	// d_res[res_idx + 1] = maxW - minW;
	// d_res[res_idx + 2] = a_idx;
	// d_res[res_idx + 3] = b_idx;
}


int* CudaCombine::createDeviceAdjArray(const std::vector<int>& allReqAdj){
	int* d_adj;
	const size_t mem_size = __SIZE_ADJ * sizeof(int);
	checkCudaErrors(cudaMalloc((void **)&d_adj, mem_size));

	int* h_adj = (int*)(allReqAdj.data());
	checkCudaErrors(cudaMemcpy(d_adj, h_adj, mem_size, cudaMemcpyHostToDevice));

	std::cout << "mem size adj: " << mem_size << ", (MB): " << ((float)mem_size)/1024.0/1024.0 << ", pointer: " << d_adj << std::endl;
	return d_adj;
}


int* CudaCombine::createDeviceConnArray(const std::vector<int>& conns){
	int* d_conn;
	const size_t mem_size = conns.size() * sizeof(int);
	checkCudaErrors(cudaMalloc((void **)&d_conn, mem_size));

	int* h_conn = (int*)(conns.data());
	checkCudaErrors(cudaMemcpy(d_conn, h_conn, mem_size, cudaMemcpyHostToDevice));

	std::cout << "mem size conn: " << mem_size << ", (MB): " << ((float)mem_size)/1024.0/1024.0 << ", pointer: " << d_conn << std::endl;
	return d_conn;
}


int16_t* CudaCombine::createDeviceCoreLayoutsArray(const std::vector<int16_t>& pts){
	int16_t* d_pts;
	const size_t mem_size = pts.size() * sizeof(int16_t);
	checkCudaErrors(cudaMalloc((void **)&d_pts, mem_size));

	int16_t* h_pts = (int16_t*)(pts.data());
	checkCudaErrors(cudaMemcpy(d_pts, h_pts, mem_size, cudaMemcpyHostToDevice));

	std::cout << "mem size core layout: " << mem_size << ", (MB): " << ((float)mem_size)/1024.0/1024.0 << ", pointer: " << d_pts << std::endl;
	return d_pts;
}

int* CudaCombine::createDeviceResArray(const size_t mem_size) {
	int *d_res = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_res, mem_size));
	checkCudaErrors(cudaMemset(d_res, -1, mem_size));

	std::cout << "mem size res: " << mem_size << ", (MB): " << ((float)mem_size)/1024.0/1024.0 << ", pointer: " << d_res << std::endl;
	return d_res;
}

void CudaCombine::freeDeviceArrays(int* adj, int* res, int* conn, int16_t* a, int16_t* b) {
	checkCudaErrors(cudaFree(a));
	checkCudaErrors(cudaFree(b));
	checkCudaErrors(cudaFree(adj));
	checkCudaErrors(cudaFree(res));
	checkCudaErrors(cudaFree(conn));
}

void CudaCombine::createPts(
		const size_t res_mem_size,
		const long NConn,
		const long num_a,
		const long qtd_b,
		const long a_offset,
		const long num_blocks,
		const long num_threads,
		int* h_res,
		int* d_adj,
		int* d_res,
		int* d_conns,
		int16_t* d_a,
		int16_t* d_b) 
	{
	dim3 grid(num_blocks, num_a, NConn);
	dim3 threads(num_threads, 1, 1);

	checkCudaErrors(cudaMemset(d_res, -1, res_mem_size));

	k_createPts<<<grid, threads>>>(d_a, d_b, d_res, d_adj, d_conns, num_a, qtd_b, a_offset);
	cudaDeviceSynchronize();

	checkCudaErrors(cudaMemcpy(h_res, d_res, res_mem_size, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
}