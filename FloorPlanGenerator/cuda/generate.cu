#include <cstdio>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <algorithm>
#include <filesystem>
#include <fstream>

#include "helper.cuh"
#include "generate.cuh"
#include "process.h"
#include "common.cuh"
#include "../lib/log.h"
#include "../lib/cvHelper.h"
#include "../lib/globals.h"
#include "../lib/calculator.h"

__global__
void generate(
	int *d_rooms_config, 
	int *d_perm, 
	int *d_adj,
	int *d_adj_count,
	int16_t *d_res, 
	const long size_idx_offset, 
	const long max_size_idx)
{
	int conn_idx = blockIdx.y;
	const int perm_idx = blockIdx.z  * __GENERATE_N;
	const int rotation_idx = threadIdx.x;
	long size_idx = (blockIdx.x * blockDim.y) + threadIdx.y;
	// long size_idx_temp = (blockIdx.x * blockDim.y) + threadIdx.y;
	const unsigned long res_idx = ((blockIdx.z * gridDim.y * max_size_idx * blockDim.x) + (blockIdx.y * max_size_idx * blockDim.x) + (blockIdx.x * blockDim.x * blockDim.y)  + (threadIdx.y * blockDim.x) + threadIdx.x) * (long)__GENERATE_RES_LENGHT;

	__shared__ int rooms_config[__GENERATE_N * __ROOM_CONFIG_LENGHT];
	if(threadIdx.y < (__GENERATE_N * __ROOM_CONFIG_LENGHT) && threadIdx.x == 0){
		rooms_config[threadIdx.y] = d_rooms_config[threadIdx.y];
	}

	__shared__ int perm[__GENERATE_N];
	if(threadIdx.y < __GENERATE_N && threadIdx.x == 0){
		perm[threadIdx.y] = d_perm[threadIdx.y + perm_idx];
	}

	__shared__ int adj_count[__SIZE_ADJ_TYPES];
	if(threadIdx.y < __SIZE_ADJ_TYPES && threadIdx.x == 0){
		adj_count[threadIdx.y] = d_adj_count[threadIdx.y + perm_idx];
	}

	__shared__ int req_adj[__SIZE_ADJ];
	if(threadIdx.y < __SIZE_ADJ && threadIdx.x == 0){
		req_adj[threadIdx.y] = d_adj[threadIdx.y];
	}

	int result[__GENERATE_RES_LAYOUT_LENGHT];
	for(int i = 0; i < __GENERATE_RES_LAYOUT_LENGHT; i++){
		result[i] = 0;
	}

	__syncthreads();

	if(size_idx >= max_size_idx)
		return;			

	size_idx += size_idx_offset;

	for(int i = 0; i < __GENERATE_N; i++){
		const int id = perm[i];
		const int offset_idx = (i * 4) + 2;
		const int room_idx = id * __ROOM_CONFIG_LENGHT;
		const int step = rooms_config[room_idx + __ROOM_CONFIG_STEP];
		const int minH = rooms_config[room_idx + __ROOM_CONFIG_MINH];
		const int maxH = rooms_config[room_idx + __ROOM_CONFIG_MAXH];
		const int minW = rooms_config[room_idx + __ROOM_CONFIG_MINW];
		const int maxW = rooms_config[room_idx + __ROOM_CONFIG_MAXW];
		const int countH = rooms_config[room_idx + __ROOM_CONFIG_COUNTH];
		const int countW = rooms_config[room_idx + __ROOM_CONFIG_COUNTW];

		int h = ((size_idx % countH) * step) + minH;
		if(h > maxH){
			h = maxH;
		}
		size_idx /= countH;

		int w = ((size_idx % countW) * step) + minW;
		if(w > maxW){
			w = maxW;
		}
		size_idx /= countW;

		if(rotation_idx & (1 << id)){
			if(w == h)
				return;

			result[offset_idx] = h;
			result[offset_idx + 1] = w;
		} else {
			result[offset_idx] = w;
			result[offset_idx + 1] = h;
		}
	}

	for(int i = 1; i < __GENERATE_N; i++){
		const int connections = 3*i*4;
		const int res_offset = i*4;
		int conn = conn_idx % connections;
		conn_idx /= connections;

		conn += (conn/4) + (conn/12) + 1;
		int srcConn = conn >> 2;
		int dstConn = conn & 3;

		if(dstConn == srcConn)
			return;

		const int srcW = result[(srcConn & ~3) | ((srcConn & 1) << 1)];
		const int srcH = result[srcConn | 1];
		
		dstConn += res_offset;
		const int dstW = result[(dstConn & ~3) | ((dstConn & 1) << 1)];
		const int dstH = result[(dstConn | 1)];

		const int diffW = srcW - dstW;
		const int diffH = srcH - dstH;

		result[res_offset] += diffW;
		result[res_offset + 1] += diffH;
		result[res_offset + 2] += diffW;
		result[res_offset + 3] += diffH;
	}

 	int connections[__GENERATE_N];
	for(int i = 0; i < __GENERATE_N; i++){
		connections[i] = 1 << i;
	}

	for(int i = 0; i < __GENERATE_RES_LAYOUT_LENGHT; i+=4){
		const int a_left = result[i];
		const int a_up = result[i + __UP];
		const int a_right = result[i + __RIGHT];
		const int a_down = result[i + __DOWN];

		for(int j = i + 4; j < __GENERATE_RES_LAYOUT_LENGHT; j+=4){
			const int b_left = result[j];
			const int b_up = result[j + __UP];
			const int b_right = result[j + __RIGHT];
			const int b_down = result[j + __DOWN];

			if(!check_overlap(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right))
				return;

			if(check_adjacency(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right)){
				connections[i/4] |= 1 << (j/4);
				connections[j/4] |= 1 << (i/4); 
			}
		}
	}

	int adj[__SIZE_ADJ_TYPES]; //Rid connections from the specific rId
	for(int i = 0; i < __SIZE_ADJ_TYPES; i++){
		adj[i] = 0;
	}

	int layout_rids = 0;
	for(int i = 0; i < __GENERATE_N; i++){
		const int id = perm[i];
		const int rid = rooms_config[id * __ROOM_CONFIG_LENGHT + __ROOM_CONFIG_RID];
		adj[rid] |= connections[i];
		layout_rids |= rid << (i * __PERM_BITS_SIZE);
	}

	// if(res_idx > 1910000 && res_idx < 1940000){
	// if(res_idx  == 104){
	// 	printf("%ld\nperm_idx: %d, (%d, %d, %d)\nrids : %d, %d, %d\nbx: %d, by: %d, bz: %d, tx: %d, ty: %d, tz: %d\nconn: %d, %d, %d\nadj_count: %d, %d, %d, %d\nadj: %d, %d, %d, %d\n\n",
	// 			res_idx, 
	// 			perm_idx, perm[perm_idx + 0], perm[perm_idx + 1], perm[perm_idx + 2],
	// 			rooms_config[perm[perm_idx + 0] * __ROOM_CONFIG_LENGHT + __ROOM_CONFIG_RID], rooms_config[perm[perm_idx + 1] * __ROOM_CONFIG_LENGHT + __ROOM_CONFIG_RID], rooms_config[perm[perm_idx + 2] * __ROOM_CONFIG_LENGHT + __ROOM_CONFIG_RID],
	// 			blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,
	// 			connections[0], connections[1], connections[2],
	// 			adj_count[0], adj_count[1], adj_count[2], adj_count[3],
	// 			adj[0], adj[1], adj[2], adj[3]
	// 	);
	// 	printf("result: \n(%d, %d), (%d, %d)\n(%d, %d), (%d, %d)\n(%d, %d), (%d, %d)\n\n",
	// 			result[0], result[1], result[2], result[3],
	// 			result[4], result[5], result[6], result[7],
	// 			result[8], result[9], result[10], result[11]
	// 	);
	// }
	
	for(int i = 0; i < __SIZE_ADJ_TYPES; i++){
		for(int j = 0; j < __SIZE_ADJ_TYPES; j++){
			const int req_adj_idx = i*__SIZE_ADJ_TYPES + j;
			if(req_adj[req_adj_idx] == REQ_ANY && !(adj[j] & adj_count[i]))
				return;

			if(req_adj[req_adj_idx] == REQ_ALL && (adj[j] & adj_count[i]) != adj_count[i])
				return;

			// if(req_adj[req_adj_idx] == REQ_ANY && !(adj[j] & adj_count[i]))
			// 	layout_rids = -3;

			// if(req_adj[req_adj_idx] == REQ_ALL && (adj[j] & adj_count[i]) != adj_count[i])
			// 	layout_rids = -2;
			
		}
	}

	// if(res_idx > (673920000 - __GENERATE_RES_LAYOUT_LENGHT)){
	// 	printf("%ld, size_idx: %ld, max_size_idx: %d\nbx: %d, by: %d, bz: %d, tx: %d, ty: %d, tz: %d\nthreadIdx.x: %ld\n(threadIdx.y * blockDim.x): %ld\n(blockIdx.x * blockDim.x * blockDim.y): %ld\n(blockIdx.y * gridDim.x * blockDim.x * blockDim.y): %ld\n(blockIdx.z * gridDim.y * gridDim.x * blockDim.x * blockDim.y):%ld\n\n",
	// 			res_idx, size_idx_temp, max_size_idx,
	// 			blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,
	// 			threadIdx.x,
	// 			(threadIdx.y * blockDim.x),
	// 			(blockIdx.x * blockDim.x * blockDim.y),
	// 			(blockIdx.y * max_size_idx * blockDim.y),
	// 			(blockIdx.z * gridDim.y * max_size_idx * blockDim.y)
	// 	);
	// }

	for(int i = 0; i < __GENERATE_RES_LAYOUT_LENGHT; i++){
		d_res[res_idx + i] = result[i];
	}

	d_res[res_idx + __GENERATE_RES_LAYOUT_LENGHT] = layout_rids;
}


__global__
void checkDuplicates2(
	int16_t *d_res, 
	const long res_a,
	const long max_layout_idx)
{
	// const long res_a = blockIdx.x;
	const long res_b = ((blockIdx.x * blockDim.x) + threadIdx.x) + res_a + 1;

	if(res_a >= max_layout_idx || res_b >= max_layout_idx)
		return;		

	const long offset_a = res_a * __GENERATE_RES_LENGHT;
	const long offset_b = res_b * __GENERATE_RES_LENGHT;

	__shared__ int layout_a[__GENERATE_RES_LAYOUT_LENGHT];
	if(threadIdx.x < __GENERATE_RES_LAYOUT_LENGHT){
		layout_a[threadIdx.x] = d_res[offset_a + threadIdx.x];
	}

	int layout_b[__GENERATE_RES_LAYOUT_LENGHT];
	for(int i = 0; i < __GENERATE_RES_LAYOUT_LENGHT; i++){
		layout_b[i] = d_res[offset_b + i];
	}

	__syncthreads();

	if(layout_a[0] == -1 || layout_b[0] == -1)
		return;

	for(int i = 0; i < __GENERATE_RES_LAYOUT_LENGHT; i++){
		if(layout_a[i] != layout_b[i])
			return;
	}

	d_res[offset_b]	= -1;
}

int* CudaGenerate::createDeviceRoomConfigsArray(const std::vector<RoomConfig>& rooms){
	const long configs_mem_size = __GENERATE_N * __ROOM_CONFIG_LENGHT * sizeof(int);
	
	int *h_configs = nullptr;
	// h_configs = (int*)malloc(configs_mem_size);
	cudaMallocHost((void**)&h_configs, configs_mem_size);	
	memset(h_configs, 0, configs_mem_size);
	
	for(int i = 0; i < __GENERATE_N; i++){
		const int offset = i * __ROOM_CONFIG_LENGHT;
		h_configs[offset + __ROOM_CONFIG_STEP] = rooms[i].step;
		h_configs[offset + __ROOM_CONFIG_MINH] = rooms[i].minH;
		h_configs[offset + __ROOM_CONFIG_MAXH] = rooms[i].maxH;
		h_configs[offset + __ROOM_CONFIG_MINW] = rooms[i].minW;
		h_configs[offset + __ROOM_CONFIG_MAXW] = rooms[i].maxW;

		const int countH = (((rooms[i].maxH - rooms[i].minH) + rooms[i].step - 1) / rooms[i].step) + 1;
		const int countW = (((rooms[i].maxW - rooms[i].minW) + rooms[i].step - 1) / rooms[i].step) + 1;
		h_configs[offset + __ROOM_CONFIG_COUNTH] = countH;
		h_configs[offset + __ROOM_CONFIG_COUNTW] = countW;

		h_configs[offset + __ROOM_CONFIG_RID] = rooms[i].rPlannyId;
	}

	int *d_configs = nullptr;
	checkCudaErrors(cudaMalloc((void **)&d_configs, configs_mem_size));
	checkCudaErrors(cudaMemcpy(d_configs, h_configs, configs_mem_size, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();	

	checkCudaErrors(cudaFreeHost(h_configs));
	// free(h_configs);
	return d_configs;
}

int* CudaGenerate::createDevicePermArray(){
	const long perm_mem_size = __GENERATE_N * __GENERATE_PERM * sizeof(int);

	int *h_perm = nullptr;
	cudaMallocHost((void**)&h_perm, perm_mem_size);	
	memset(h_perm, 0, perm_mem_size);
	
	std::vector<int> perm;
	for(int i = 0; i < __GENERATE_N; i++){
		perm.push_back(i);
	}

	int idx = 0;
	do {
		for(int i = 0; i < __GENERATE_N; i++){
			h_perm[(idx * __GENERATE_N) + i] = perm[i];
		}
		idx++;
	} while (std::next_permutation(perm.begin(), perm.end()));

	int *d_perm = nullptr;
	checkCudaErrors(cudaMalloc((void **)&d_perm, perm_mem_size));
	checkCudaErrors(cudaMemcpy(d_perm, h_perm, perm_mem_size, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();	

	checkCudaErrors(cudaFreeHost(h_perm));
	return d_perm;
}

int* CudaGenerate::createDeviceAdjArray(
	const std::vector<RoomConfig>& rooms, 
	std::vector<int> allReq, 
	std::vector<int> reqCount)
{
	std::vector<int> originalReqCount = reqCount;

    for(const RoomConfig room : rooms){
        reqCount[room.rPlannyId] -= 1;
    }

    for(int i = 0; i < __SIZE_ADJ_TYPES; i++){
        for(int j = 0; j < __SIZE_ADJ_TYPES; j++){
			int idx_i = i*__SIZE_ADJ_TYPES + j;
			int idx_j = j*__SIZE_ADJ_TYPES + i;

			if((reqCount[i] == originalReqCount[i] || reqCount[j] == originalReqCount[j]) || 
			   (i == j && reqCount[i] == 1))
			{
				allReq[idx_i] = REQ_NONE;
				allReq[idx_j] = REQ_NONE;
			}
			else if(allReq[idx_i] == REQ_ANY && (reqCount[i] > 0 || reqCount[j] > 0)){
				allReq[idx_i] = REQ_NONE;
			}
			else if(allReq[idx_i] == REQ_ALL && (reqCount[j] > 0)){
				allReq[idx_i] = REQ_NONE;
			}
        }
    }

	// std::cout << "adj:" << std::endl;
    // for(int i = 0; i < __SIZE_ADJ_TYPES; i++){
    //     for(int j = 0; j < __SIZE_ADJ_TYPES; j++){
	// 		std::cout << allReq[i * __SIZE_ADJ_TYPES + j] << ", ";
	// 	}	
	// 	std::cout << std::endl;
	// }
	// std::cout << std::endl;

	int *h_adj = (int *)(&allReq[0]);
	const unsigned long mem_size_adj = sizeof(int) * __SIZE_ADJ;
	
	int *d_adj = nullptr;
	checkCudaErrors(cudaMalloc((void **)&d_adj, mem_size_adj));
	checkCudaErrors(cudaMemcpy(d_adj, h_adj, mem_size_adj, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();	

	return d_adj;
}

int* CudaGenerate::createDeviceAdjCountArray(const std::vector<RoomConfig>& rooms) {
	const long mem_size_adj_count = __SIZE_ADJ_TYPES * __GENERATE_PERM * sizeof(int);

	int *h_adj_count = nullptr;
	cudaMallocHost((void**)&h_adj_count, mem_size_adj_count);	
	memset(h_adj_count, 0, mem_size_adj_count);
	
	std::vector<int> perm;
	for(int i = 0; i < __GENERATE_N; i++){
		perm.push_back(i);
	}

	int idx = 0;
	do {
		for(int i = 0; i < __GENERATE_N; i++){
			int offset = idx * __SIZE_ADJ_TYPES;
			int rid = rooms[perm[i]].rPlannyId;
			h_adj_count[offset + rid] |= 1 << i;

			// std::cout << "idx: " << idx << ", i: " << i << ", rid: " << rid << std::endl;
		}

		// std::cout << "perm: ";
		// for(int i = 0; i < __GENERATE_N; i++){
		// 	std::cout << perm[i] << ", ";
		// }
		// std::cout << std::endl;

		// std::cout << "h_adj_count: ";
		// for(int i = 0; i < __SIZE_ADJ_TYPES; i++){
		// 	std::cout << h_adj_count[(idx * __SIZE_ADJ_TYPES) + i] << ", ";
		// }
		// std::cout << std::endl;

		idx++;
	} while (std::next_permutation(perm.begin(), perm.end()));

	
	int *d_adj_count = nullptr;
	checkCudaErrors(cudaMalloc((void **)&d_adj_count, mem_size_adj_count));
	checkCudaErrors(cudaMemcpy(d_adj_count, h_adj_count, mem_size_adj_count, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();	

	checkCudaErrors(cudaFreeHost(h_adj_count));
	return d_adj_count;
}

int16_t* CudaGenerate::createDeviceResArray(const size_t result_mem_size) {
	int16_t *d_res = nullptr;

	checkCudaErrors(cudaMalloc((void**)&d_res, result_mem_size));
	std::cout << "d_res: " << d_res << std::endl;
	return d_res;
}

// int16_t** CudaGenerate::createHostResArray(const size_t result_mem_size, const int nThreads) {
// 	int16_t** h_res = (int16_t**)calloc(nThreads, sizeof(int16_t*));

// 	for(int i = 0; i < nThreads; i++){
// 		checkCudaErrors(cudaMallocHost((void**)(&(h_res[i])), result_mem_size));
// 	}
// 	return h_res;
// }

void CudaGenerate::launchGenereteKernel(
	const int qtdBlocksX, 
	const int qtdThreadY, 
	const long NConn, 
	const long NPerm, 
	const long qtdSizes, 
	int* d_configs, 
	int* d_perm, 
	int* d_adj, 
	int* d_adj_count, 
	int16_t* d_res, 
	int16_t* h_res, 
	const long size_idx_offset,
	const size_t result_mem_size)
{
	dim3 grid(qtdBlocksX, NConn, NPerm);
	dim3 threads(__GENERATE_ROTATIONS, qtdThreadY, 1);

	checkCudaErrors(cudaMemset(d_res, -1, result_mem_size));

	generate<<<grid, threads>>>(d_configs, d_perm, d_adj, d_adj_count, d_res, size_idx_offset, qtdSizes);

	// memset(h_res, -1, result_mem_size);
	cudaDeviceSynchronize();	

	checkCudaErrors(cudaMemcpy(h_res, d_res, result_mem_size, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	int found = 0;
	for(int i = 0; i < result_mem_size / (sizeof(int16_t)); i+= __GENERATE_RES_LENGHT){
		found += (h_res[i] == -1) ? 0 : 1;
	}
	if(!found){
		std::cout << "not found" << std::endl;
	}
}

void CudaGenerate::launchDuplicateCheckKernel(
	int16_t* d_res, 
	int16_t* h_res,
	const long layouts_count,
	const size_t result_mem_size)
{
	for(long i = 0; i < layouts_count - 1; i++){
		long offset_a = i * __GENERATE_RES_LENGHT;

		// if(i % 1000 == 0){
		// 	std::cout << "launchDuplicateCheckKernel " << i  << std::endl;
		// }
		if(h_res[offset_a] == -1)
			continue;

		long layouts_count_b = layouts_count - i - 1;
		long threadX = layouts_count_b > 768 ? 768 : layouts_count_b;
		long blockX = (layouts_count_b + threadX - 1) / threadX;

		dim3 grid(blockX, 1, 1);
		dim3 threads(threadX, 1, 1);
		
		checkDuplicates2<<<grid, threads>>>(d_res, i, layouts_count);
	}
	cudaDeviceSynchronize();	

	checkCudaErrors(cudaMemcpy(h_res, d_res, result_mem_size, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
}

void CudaGenerate::freeDeviceArrays(int* d_configs, int* d_perm, int* d_adj, int* d_adj_count, int16_t* d_res){
	checkCudaErrors(cudaFree(d_configs));
	checkCudaErrors(cudaFree(d_perm));
	checkCudaErrors(cudaFree(d_adj));
	checkCudaErrors(cudaFree(d_adj_count));
	checkCudaErrors(cudaFree(d_res));
}

void CudaGenerate::freeHostArrays(int16_t** h_res, const int nThreads){
	for(int i = 0; i < nThreads; i++){
		checkCudaErrors(cudaFreeHost(h_res[i]));
	}

	free(h_res);
}