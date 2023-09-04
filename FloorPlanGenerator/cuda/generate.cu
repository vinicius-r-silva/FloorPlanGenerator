#include <cstdio>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <algorithm>
#include <filesystem>
#include <fstream>

#include "helper.cuh"
#include "generate.h"
#include "process.h"
#include "common.cuh"
#include "../lib/log.h"
#include "../lib/cvHelper.h"
#include "../lib/globals.h"
#include "../lib/calculator.h"

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
	const long res_idx = ((blockIdx.z * gridDim.y * gridDim.x * blockDim.x * blockDim.y) + (blockIdx.y * gridDim.x * blockDim.x * blockDim.y) + (blockIdx.x * blockDim.x * blockDim.y)  + (threadIdx.y * blockDim.x) + threadIdx.x) * (long)__GENERATE_RES_LENGHT;

	if(size_idx > max_size_idx)
		return;


	size_idx += size_idx_offset;

	__shared__ int rooms_config[__GENERATE_N * __ROOM_CONFIG_LENGHT];
	if(threadIdx.y < (__GENERATE_N * __ROOM_CONFIG_LENGHT) && threadIdx.x == 0){
		rooms_config[threadIdx.y] = d_rooms_config[threadIdx.y];
	}

	__shared__ int perm[__GENERATE_N];
	if(threadIdx.y < __GENERATE_N && threadIdx.x == 0){
		perm[threadIdx.y] = d_perm[threadIdx.y + perm_idx];
	}

	__shared__ int adj_count[__GENERATE_N];
	if(threadIdx.y < __GENERATE_N && threadIdx.x == 0){
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

	if(size_idx > 0 || rotation_idx > 0)
		return;

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

	for(int i = 0; i < __GENERATE_N; i++){
		const int id = perm[i];
		const int rid = rooms_config[id * __ROOM_CONFIG_LENGHT + __ROOM_CONFIG_RID];
		adj[rid] |= connections[i];
	}

	// if(res_idx > 1910000 && res_idx < 1940000){
	// if(res_idx > 0 && res_idx < 40000){
	// 	printf("%ld\nperm_idx: %d, (%d, %d, %d)\nrids : %d, %d, %d\nbx: %d, by: %d, bz: %d, tx: %d, ty: %d, tz: %d\nconn: %d, %d, %d\nadj_count: %d, %d, %d, %d\nadj: %d, %d, %d, %d\n\n",
	// 			res_idx, 
	// 			perm_idx, perm[perm_idx + 0], perm[perm_idx + 1], perm[perm_idx + 2],
	// 			rooms_config[perm[perm_idx + 0] * __ROOM_CONFIG_LENGHT + __ROOM_CONFIG_RID], rooms_config[perm[perm_idx + 1] * __ROOM_CONFIG_LENGHT + __ROOM_CONFIG_RID], rooms_config[perm[perm_idx + 2] * __ROOM_CONFIG_LENGHT + __ROOM_CONFIG_RID],
	// 			blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,
	// 			connections[0], connections[1], connections[2],
	// 			adj_count[0], adj_count[1], adj_count[2], adj_count[3],
	// 			adj[0], adj[1], adj[2], adj[3]
	// 	);
	// }
	
	for(int i = 0; i < __SIZE_ADJ_TYPES; i++){
		for(int j = 0; j < __SIZE_ADJ_TYPES; j++){
			const int req_adj_idx = i*__SIZE_ADJ_TYPES + j;
			if(req_adj[req_adj_idx] == REQ_ANY && !(adj[j] & adj_count[i]))
				return;

			if(req_adj[req_adj_idx] == REQ_ALL && (adj[j] & adj_count[i]) != adj_count[i])
				return;
		}
	}

	for(int i = 0; i < __GENERATE_RES_LAYOUT_LENGHT; i++){
		d_res[res_idx + i] = result[i];
	}

	// //TODO replace to adj idx permuated
	d_res[res_idx + __GENERATE_RES_LAYOUT_LENGHT] =  blockIdx.y;

}

int* CudaGenerate::createDeviceRoomConfigsArray(const std::vector<RoomConfig>& rooms){
	const long configs_mem_size = __GENERATE_N * __ROOM_CONFIG_LENGHT * sizeof(int);
	
	int *h_configs = nullptr;
	cudaMallocHost((void**)&h_configs, configs_mem_size);	
	
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
	return d_configs;
}

int* CudaGenerate::createDevicePermArray(){
	const long perm_mem_size = __GENERATE_N * __GENERATE_PERM * sizeof(int);

	int *h_perm = nullptr;
	cudaMallocHost((void**)&h_perm, perm_mem_size);	
	
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
	std::vector<int> reqCount,
	const int reqSize)
{
	std::vector<int> originalReqCount = reqCount;

    for(const RoomConfig room : rooms){
        reqCount[room.rPlannyId] -= 1;
    }

    for(int i = 0; i < reqSize; i++){
        for(int j = 0; j < reqSize; j++){
			int idx_i = i*reqSize + j;
			int idx_j = j*reqSize + i;


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
    // for(int i = 0; i < reqSize; i++){
    //     for(int j = 0; j < reqSize; j++){
	// 		std::cout << allReq[i * reqSize + j] << ", ";
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

	return d_adj_count;
}
 
void CudaGenerate::generateCuda(
	const std::vector<RoomConfig>& rooms, 
	std::vector<int>& allReq, 
	std::vector<int> allReqCount,
	const int reqSize)
{
	if(rooms.size() != __GENERATE_N)
		return;

	std::cout << std::endl << std::endl << std::endl;
	for(int i = 0; i < __GENERATE_N; i++){
		Log::print(rooms[i]);
	}

	// const long targetMemSize = (45l * 1024l * 1024l * 1024l) / 10l;
	const long targetMemSize = 8l * 1024l * 1024l * 1024l;

	// int k = 2;
	// for(int i = 0; i < 3*k*4; i++){
	// 	int conn_idx = i;
	// 	std::cout << conn_idx << ": "; 

	// 	const int connections = 3*k*4;

	// 	int conn = conn_idx % connections;
	// 	conn_idx /= connections;

	// 	int newconn = conn + (conn/4) + (conn/12) + 1;
	// 	int srcConn = newconn >> 2;
	// 	int dstConn = newconn & 3;



	// 	const int srcW_idx = (srcConn & ~3) | ((srcConn & 1) << 1);
	// 	const int srcH_idx = srcConn | 1;
		
	// 	dstConn += k * 4;
	// 	const int dstW_idx = (dstConn & ~3) | ((dstConn & 1) << 1);
	// 	const int dstH_idx = (dstConn | 1);

	// 	std::cout  << "  (" << srcConn << ", " << dstConn << "), " << "conn: " << conn << ", newconn: " << newconn << ", srcW_idx: " << srcW_idx << ", srcH_idx: " << srcH_idx << ", dstW_idx: " << dstW_idx << ", dstH_idx: " << dstH_idx << std::endl;
	// }
	// return;

	long NSizes = 1;
    for(const RoomConfig room : rooms){
		NSizes *= (((room.maxH - room.minH + room.step - 1) / room.step) + 1) * (((room.maxW - room.minW + room.step - 1) / room.step) + 1);
    }

    const long NConn = Calculator::NConnectionsReduced(__GENERATE_N);
    const long NPerm = Calculator::Factorial(__GENERATE_N);
    const long NSizesRotation = NSizes * __GENERATE_ROTATIONS;

	std::cout << "NConn: " << NConn << ", NPerm: " << NPerm << std::endl;
	std::cout << "NSizes: " << NSizes << ", NSizesRotation: " << NSizesRotation << std::endl;

	const int targetThreadsPerBlock = 768;
	const int targetQtdThreadsX = targetThreadsPerBlock / __GENERATE_ROTATIONS;
	if(targetThreadsPerBlock % __GENERATE_ROTATIONS != 0){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!! make the targetThreadsPerBlock divisible by " << __GENERATE_ROTATIONS << "!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return; 
	}

	if(targetThreadsPerBlock < __GENERATE_N * __ROOM_CONFIG_LENGHT){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!! not enought threads to fill config array !!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return; 
	}

	if(targetThreadsPerBlock < __SIZE_ADJ){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!! not enought threads to fill adj array !!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return; 
	}

	if(reqSize * reqSize != __SIZE_ADJ){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!! wrong __SIZE_ADJ value !!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return; 
	}

	if(reqSize != __SIZE_ADJ_TYPES){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!! wrong __SIZE_ADJ_TYPES value !!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return; 
	}

	

	const long maxLayoutsPerKernel = targetMemSize / (__GENERATE_RES_LENGHT * sizeof(int16_t));
	const long maxQtdSizes = (maxLayoutsPerKernel / (NConn * NPerm * targetQtdThreadsX * __GENERATE_ROTATIONS)) * targetQtdThreadsX;
	const long qtdSizes = maxQtdSizes < NSizes ? maxQtdSizes : NSizes;
	const long layoutsPerKernel = qtdSizes * NConn * NPerm * __GENERATE_ROTATIONS;

	std::cout << "maxLayoutsPerKernel: " << maxLayoutsPerKernel << std::endl;
	std::cout << "qtdSingleSize: " << NConn * NPerm << std::endl;
	std::cout << "layoutsPerKernel: " << layoutsPerKernel << std::endl;
	std::cout << "maxQtdSizes: " << maxQtdSizes << ", qtdSizes: " << qtdSizes << std::endl;
	std::cout << "kernel launchs: " << NConn * NPerm * (qtdSizes / targetQtdThreadsX)  << std::endl;

	int* d_configs = CudaGenerate::createDeviceRoomConfigsArray(rooms);
	int* d_perm = CudaGenerate::createDevicePermArray();
	int* d_adj = CudaGenerate::createDeviceAdjArray(rooms, allReq, allReqCount, reqSize);
	int* d_adj_count = CudaGenerate::createDeviceAdjCountArray(rooms);
	// return;

	int16_t *d_res = nullptr;
	const long result_mem_size = qtdSizes * NConn * NPerm * __GENERATE_ROTATIONS * __GENERATE_RES_LENGHT * sizeof(int16_t);

	cudaMalloc((void**)&d_res, result_mem_size);	
	checkCudaErrors(cudaMemset(d_res, -1, result_mem_size));

	const int qtdThreadY = qtdSizes > targetQtdThreadsX ? targetQtdThreadsX : qtdSizes;
	const int qtdBlocksX = (qtdSizes + qtdThreadY - 1) / qtdThreadY;

	dim3 grid(qtdBlocksX, NConn, NPerm);
	dim3 threads(__GENERATE_ROTATIONS, qtdThreadY, 1);

	std::cout << "result_mem_size: " << result_mem_size << std::endl;
	std::cout << "targetThreadsPerBlock: " << targetThreadsPerBlock << ", targetQtdThreadsX: " << targetQtdThreadsX << std::endl;
	std::cout << "qtdThreadY: " << qtdThreadY << ", qtdBlocksX: " << qtdBlocksX << std::endl;
	std::cout << "grid: " << grid.x << ", " << grid.y << ", " << grid.z << std::endl;
	std::cout << "threads: " << threads.x << ", " << threads.y << ", " << threads.z << std::endl;

	generate<<<grid, threads>>>(d_configs, d_perm, d_adj, d_adj_count, d_res, 0, qtdSizes);
	cudaDeviceSynchronize();	
	// for(int i = 0; i < NSizes; i+= qtdSizes){
	// 	int diff = NSizes - i;

	// 	if(diff < qtdSizes){
	// 		generate<<<grid, threads>>>(d_configs, d_res, i, diff);
	// 		cudaDeviceSynchronize();	
	// 	} else {
	// 		generate<<<grid, threads>>>(d_configs, d_res, i, qtdSizes);
	// 		cudaDeviceSynchronize();	
	// 	}
	// }

	int16_t *h_res = nullptr;
	cudaMallocHost((void**)&h_res, result_mem_size);	
	checkCudaErrors(cudaMemcpy(h_res, d_res, result_mem_size, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();	

	// // for(int i = 0; i < layoutsPerKernel; i++){
	// for(int i = 0; i < layoutsPerKernel; i+= qtdBlocksX * qtdThreadY * __GENERATE_ROTATIONS ){
	// // for(int i = 0; i < layoutsPerKernel; i+= qtdBlocksX * qtdThreadY * NConn * __GENERATE_ROTATIONS){
	// 	if(h_res[(i * __GENERATE_RES_LENGHT) + 2] == 0)
	// 		continue;

	// 	std::cout << i * __GENERATE_RES_LENGHT << ":  ";
	// 	for(int j = 0; j < __GENERATE_RES_LENGHT; j++){
	// 		std::cout << (int)h_res[(i * __GENERATE_RES_LENGHT) + j] << ", ";
	// 	}
	// 	std::cout << std::endl;
	// 	getchar();
	// }
	// std::cout << std::endl;


	std::vector<int16_t> result_vector;
	for(int i = 0; i < layoutsPerKernel; i++){
	// for(int i = 0; i < layoutsPerKernel; i+= qtdBlocksX * qtdThreadY * __GENERATE_ROTATIONS){
		if(h_res[(i * __GENERATE_RES_LENGHT)] == -1)
			continue;

		for(int j = 0; j < __GENERATE_RES_LENGHT; j++){
			result_vector.push_back(h_res[(i * __GENERATE_RES_LENGHT) + j]);
		}
	}

	std::cout << "result size: " << result_vector.size() << ", layouts: " << result_vector.size() / __GENERATE_RES_LENGHT << std::endl;

    std::string result_data_path = "/home/ribeiro/Documents/FloorPlanGenerator/FloorPlanGenerator/storage/temp/generate.dat";
    std::ofstream outputFile(result_data_path, std::ios::out | std::ios::binary);
    outputFile.write(reinterpret_cast<const char*>(result_vector.data()), result_vector.size() * sizeof(int16_t));
    outputFile.close();

	checkCudaErrors(cudaFreeHost(h_res));

	checkCudaErrors(cudaFree(d_configs));
	checkCudaErrors(cudaFree(d_perm));
	checkCudaErrors(cudaFree(d_res));
	checkCudaErrors(cudaFree(d_adj));
	checkCudaErrors(cudaFree(d_adj_count));
}