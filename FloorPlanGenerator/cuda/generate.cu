#include <cstdio>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <algorithm>
#include <filesystem>
#include <fstream>

#include "helper.h"
#include "generate.h"
#include "process.h"
#include "../lib/log.h"
#include "../lib/cvHelper.h"
#include "../lib/globals.h"
#include "../lib/calculator.h"


// __global__
// void createSizes(int *d_rooms_config, int8_t *d_room_sizes, const long offset, const long maxIdx){
// 	long idx = blockIdx.x * blockDim.x + threadIdx.x;
// 	if(idx > maxIdx)
// 		return;
	
// 	const long res_idx = idx * __GENERATE_SIZE_LENGHT * __GENERATE_ROTATIONS;
// 	idx += offset;

// 	if(res_idx < 0){
// 		printf("aa");
// 		return;
// 	}
// 	if(idx < 0){
// 		printf("bb");
// 		return;
// 	}

// 	__shared__ int rooms_config[__GENERATE_N * __ROOM_CONFIG_LENGHT];
// 	if(threadIdx.x < __GENERATE_N * __ROOM_CONFIG_LENGHT){
// 		rooms_config[threadIdx.x] = d_rooms_config[threadIdx.x];
// 	}

// 	// //TODO remove this for sizes < 30
// 	// if(maxIdx < __GENERATE_N * __ROOM_CONFIG_LENGHT && threadIdx.x == 0){
// 	// 	for(int i = 0; i < __GENERATE_N * __ROOM_CONFIG_LENGHT; i++){
// 	// 		rooms_config[i] = d_rooms_config[i];
// 	// 		// printf("%d, ", rooms_config[i]);
// 	// 	}
// 	// 	// printf("\n");
// 	// }

// 	__syncthreads();

// 	int8_t room_sizes[__GENERATE_SIZE_LENGHT];
// 	int validRotations[__GENERATE_N];
// 	for(int i = 0; i < __GENERATE_N; i++){
// 		validRotations[i] = 0;
// 	}

// 	for(int i = 0; i < __GENERATE_N; i++){
// 		const int sizes_idx = i * 2;
// 		const int room_idx = i * __ROOM_CONFIG_LENGHT;
// 		const int step = rooms_config[room_idx + __ROOM_CONFIG_STEP];
// 		const int minH = rooms_config[room_idx + __ROOM_CONFIG_MINH];
// 		const int maxH = rooms_config[room_idx + __ROOM_CONFIG_MAXH];
// 		const int minW = rooms_config[room_idx + __ROOM_CONFIG_MINW];
// 		const int maxW = rooms_config[room_idx + __ROOM_CONFIG_MAXW];
// 		const int countH = rooms_config[room_idx + __ROOM_CONFIG_COUNTH];
// 		const int countW = rooms_config[room_idx + __ROOM_CONFIG_COUNTW];

// 		room_sizes[sizes_idx] = ((idx % countH) * step) + minH;
// 		idx /= countH;

// 		room_sizes[sizes_idx + 1] = ((idx % countW) * step) + minW;
// 		idx /= countW;

// 		validRotations[i] = room_sizes[sizes_idx] < minW || room_sizes[sizes_idx] > maxW;

// 		if(room_sizes[sizes_idx] > maxH){
// 			room_sizes[sizes_idx] = maxH;
// 		};
		
// 		if(room_sizes[sizes_idx + 1] > minW){
// 			room_sizes[sizes_idx + 1] = minW;
// 		};
// 	}

// 	for(int i = 0; i < __GENERATE_ROTATIONS; i++){
// 		int validRotation = 1;
// 		for(int j = 0; j < __GENERATE_N && validRotation; j++){
// 			if(i & (1 << j)){
// 				validRotation = validRotations[j];
// 			}
// 		}

// 		if(!validRotation)
// 			continue;


// 		const long offset = res_idx + (i * __GENERATE_SIZE_LENGHT);
// 		for(int j = 0; j < __GENERATE_N; j++){
// 			if(i & (1 << j)){
// 				d_room_sizes[offset + (j * 2)] = room_sizes[(j * 2) + 1];
// 				d_room_sizes[offset + (j * 2) + 1] = room_sizes[(j * 2)];
// 			} else {
// 				d_room_sizes[offset + (j * 2)] = room_sizes[(j * 2)];
// 				d_room_sizes[offset + (j * 2) + 1] = room_sizes[(j * 2) + 1];
// 			}
// 		}
// 	}
// }


// int8_t* CudaGenerate::generateSizes(int* d_configs, const long qtdSizes){
// 	int8_t *d_sizes = nullptr;
// 	const long sizes_mem_size = qtdSizes * __GENERATE_SIZE_LENGHT * __GENERATE_ROTATIONS * sizeof(int8_t);
// 	cudaMalloc((void**)&d_sizes, sizes_mem_size);	
// 	checkCudaErrors(cudaMemset(d_sizes, 0, sizes_mem_size));

// 	std::cout << "sizes_mem_size: " << sizes_mem_size << ", Mb: " << ((double)sizes_mem_size) / 1024.0 / 1024.0 << std::endl;

// 	const int threadsPerBlock = qtdSizes > 768 ? 768 : qtdSizes;
// 	const int qtdBlocks = (qtdSizes + threadsPerBlock - 1) / threadsPerBlock;

// 	dim3 grid(qtdBlocks, 1, 1);
// 	dim3 threads(threadsPerBlock, 1, 1);

// 	createSizes<<<grid, threads>>>(d_configs, d_sizes, 0, qtdSizes);
// 	cudaDeviceSynchronize();	

// 	// int8_t *h_sizes = nullptr;
// 	// cudaMallocHost((void**)&h_sizes, sizes_mem_size);	
// 	// checkCudaErrors(cudaMemcpy(h_sizes, d_sizes, sizes_mem_size, cudaMemcpyDeviceToHost));
// 	// cudaDeviceSynchronize();	

// 	// for(int i = 0; i < qtdSizes * __GENERATE_ROTATIONS; i++){
// 	// 	for(int j = 0; j < __GENERATE_SIZE_LENGHT; j++){
// 	// 		std::cout << (int)h_sizes[(i * __GENERATE_SIZE_LENGHT) + j] << ", ";
// 	// 	}
// 	// 	std::cout << std::endl;
// 	// }
// 	// std::cout << std::endl;

// 	return d_sizes;
// }


__global__
void generate(int *d_rooms_config, int *d_perm, int16_t *d_res, const long size_idx_offset, const long max_size_idx){
	// const int conn_idx = blockIdx.y;
	const int perm_idx = blockIdx.z;
	const int rotation_idx = threadIdx.y;
	long size_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const long res_idx = ((blockIdx.z * gridDim.y * gridDim.x * blockDim.x * blockDim.y) + (blockIdx.y * gridDim.x * blockDim.x * blockDim.y) + (blockIdx.x * blockDim.x * blockDim.y)  + (threadIdx.y * blockDim.x) + threadIdx.x) * (long)__GENERATE_RES_LENGHT;

	if(size_idx > max_size_idx)
		return;

	size_idx += size_idx_offset;

	__shared__ int rooms_config[__GENERATE_N * __ROOM_CONFIG_LENGHT];
	if(threadIdx.x < (__GENERATE_N * __ROOM_CONFIG_LENGHT) && threadIdx.y == 0){
		rooms_config[threadIdx.x] = d_rooms_config[threadIdx.x];
	}

	__shared__ int perm[__GENERATE_N * __GENERATE_PERM];
	if(threadIdx.x < (__GENERATE_N * __GENERATE_PERM) && threadIdx.y == 0){
		perm[threadIdx.x] = d_perm[threadIdx.x];
	}

	__syncthreads();


	int result[__GENERATE_RES_LAYOUT_LENGHT];
	for(int i = 0; i < __GENERATE_RES_LAYOUT_LENGHT; i++){
		result[i] = 0;
	}

	for(int i = 0; i < __GENERATE_N; i++){
		const int id = perm[(perm_idx * __GENERATE_N) + i];
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

			result[offset_idx] = w;
			result[offset_idx + 1] = h;
		} else {
			result[offset_idx] = h;
			result[offset_idx + 1] = w;
		}
	}

	for(int i = 0; i < __GENERATE_RES_LAYOUT_LENGHT; i++){
		d_res[res_idx + i] = result[i];
	}
	d_res[res_idx + __GENERATE_RES_LAYOUT_LENGHT] = perm_idx;


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
	std::cout << std::endl;


	for(int i = 0; i < __GENERATE_PERM; i++){
		std::cout << "perm " << i << ": ";
		for(int j = 0; j < __GENERATE_N; j++){
			std::cout << h_perm[(i * __GENERATE_N) + j] << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	int *d_perm = nullptr;
	checkCudaErrors(cudaMalloc((void **)&d_perm, perm_mem_size));
	checkCudaErrors(cudaMemcpy(d_perm, h_perm, perm_mem_size, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();	

	checkCudaErrors(cudaFreeHost(h_perm));
	return d_perm;
}
 
void CudaGenerate::generateCuda(const std::vector<RoomConfig>& rooms) {
	if(rooms.size() != __GENERATE_N)
		return;

	std::cout << std::endl << std::endl << std::endl;
	for(int i = 0; i < __GENERATE_N; i++){
		Log::print(rooms[i]);
	}

	// const long targetMemSize = (45l * 1024l * 1024l * 1024l) / 10l;
	const long targetMemSize = 8l * 1024l * 1024l * 1024l;

	long NSizes = 1;
    for(const RoomConfig room : rooms){
		NSizes *= (((room.maxH - room.minH + room.step - 1) / room.step) + 1) * (((room.maxW - room.minW + room.step - 1) / room.step) + 1);
    }

    const long NConn = Calculator::NConnections(__GENERATE_N);
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

	int16_t *d_res = nullptr;
	const long result_mem_size = qtdSizes * NConn * NPerm * __GENERATE_ROTATIONS * __GENERATE_RES_LENGHT * sizeof(int16_t);

	cudaMalloc((void**)&d_res, result_mem_size);	
	checkCudaErrors(cudaMemset(d_res, 0, result_mem_size));

	const int qtdThreadX = qtdSizes > targetQtdThreadsX ? targetQtdThreadsX : qtdSizes;
	const int qtdBlocksX = (qtdSizes + qtdThreadX - 1) / qtdThreadX;

	dim3 grid(qtdBlocksX, NConn, NPerm);
	dim3 threads(qtdThreadX, __GENERATE_ROTATIONS, 1);

	std::cout << "result_mem_size: " << result_mem_size << std::endl;
	std::cout << "targetThreadsPerBlock: " << targetThreadsPerBlock << ", targetQtdThreadsX: " << targetQtdThreadsX << std::endl;
	std::cout << "qtdThreadX: " << qtdThreadX << ", qtdBlocksX: " << qtdBlocksX << std::endl;
	std::cout << "grid: " << grid.x << ", " << grid.y << ", " << grid.z << std::endl;
	std::cout << "threads: " << threads.x << ", " << threads.y << ", " << threads.z << std::endl;

	generate<<<grid, threads>>>(d_configs, d_perm, d_res, 0, qtdSizes);
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

	for(int i = 0; i < layoutsPerKernel; i++){
	// for(int i = 0; i < layoutsPerKernel; i+= qtdBlocksX * qtdThreadX * NConn * __GENERATE_ROTATIONS){
		// if(h_res[(i * __GENERATE_RES_LENGHT) + 2] == 0)
		// 	continue;

		std::cout << i * __GENERATE_RES_LENGHT << ":  ";
		for(int j = 0; j < __GENERATE_RES_LENGHT; j++){
			std::cout << (int)h_res[(i * __GENERATE_RES_LENGHT) + j] << ", ";
		}
		std::cout << std::endl;
		getchar();
	}
	std::cout << std::endl;
	checkCudaErrors(cudaFreeHost(h_res));

	checkCudaErrors(cudaFree(d_configs));
	checkCudaErrors(cudaFree(d_res));
}