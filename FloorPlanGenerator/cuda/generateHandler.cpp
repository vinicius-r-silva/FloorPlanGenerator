#include <iostream>
#include <omp.h>

#include "generateHandler.h"
#include "generate.cuh"
#include "../lib/calculator.h"
#include "../lib/log.h"
#include "../lib/storage.h"


GenerateHandler::GenerateHandler(){}

void GenerateHandler::consume(std::vector<int16_t> result, Storage& hdd, const int combid, const int taskCount){
	int threadId = omp_get_thread_num();
	printf("consumer %d init, count: %d\n", threadId, taskCount);
	
	const size_t resultSize = result.size();
    // std::vector<int16_t> result_copy(result); 

	size_t dst = 0;
	for(; dst < resultSize && result[dst] != -1; dst += __GENERATE_RES_LENGHT){};

	size_t i = dst;
	size_t src_init = 0, src_end = 0;
	while(true){
	    for(; i < resultSize && result[i] == -1; i += __GENERATE_RES_LENGHT){};
		src_init = i;
		
	    for(; i < resultSize && result[i] != -1; i += __GENERATE_RES_LENGHT){};
		src_end = i;
		
		if(src_init == src_end)
			break;

		std::copy(result.begin() + src_init, result.begin() + src_end, result.begin() + dst);
		dst += src_end - src_init;

		if(i >= resultSize)
			break;
	}
	result.resize(dst);

	hdd.saveResult(result, combid, taskCount);
	printf("consumer %d end (pts: %zu, layouts: %zu)\n", threadId, result.size(), result.size() / __GENERATE_RES_LENGHT);
}

void GenerateHandler::generate(
	const std::vector<RoomConfig>& rooms, 
	std::vector<int> allReqCount,
	std::vector<int>& allReq, 
	const int reqSize,
	const int combid,
	Storage& hdd)
{
	if(GenerateHandler::checkDefineValues(rooms, reqSize))
		return;

	// std::cout << std::endl << std::endl << std::endl;
	for(int i = 0; i < __GENERATE_N; i++){
		Log::print(rooms[i]);
	}

	const size_t targetRamSize = 25l * 1024l * 1024l * 1024l;
	const size_t targetVRamSize = 8l * 1024l * 1024l * 1024l;

	long NSizes = 1;
    for(const RoomConfig room : rooms){
		long size_h = (((room.maxH - room.minH + room.step - 1) / room.step) + 1);
		long size_w = (((room.maxH - room.minH + room.step - 1) / room.step) + 1);
		std::cout << room.name << " sizes: " << size_h * size_w << std::endl;
		NSizes *= size_h * size_w;
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

	const long maxLayoutsPerKernel = targetVRamSize / (__GENERATE_RES_LENGHT * sizeof(int16_t));
	const long maxQtdSizes = (maxLayoutsPerKernel / (NConn * NPerm * targetQtdThreadsX * __GENERATE_ROTATIONS)) * targetQtdThreadsX;
	const long sizesPerLaunch = maxQtdSizes < NSizes ? maxQtdSizes : NSizes;
	const long layoutsPerKernel = sizesPerLaunch * NConn * NPerm * __GENERATE_ROTATIONS;
	const long ptsPerKernel = layoutsPerKernel * __GENERATE_RES_LENGHT;
	const size_t result_mem_size = sizesPerLaunch * NConn * NPerm * __GENERATE_ROTATIONS * __GENERATE_RES_LENGHT * sizeof(int16_t);

	std::cout << "maxLayoutsPerKernel: " << maxLayoutsPerKernel << std::endl;
	std::cout << "qtdSingleSize: " << NConn * NPerm << std::endl;
	std::cout << "layoutsPerKernel: " << layoutsPerKernel << ", ptsPerKernel: " << ptsPerKernel<< std::endl;
	std::cout << "maxQtdSizes: " << maxQtdSizes << ", sizesPerLaunch: " << sizesPerLaunch << std::endl;
	std::cout << "result_mem_size: " << result_mem_size << std::endl;
	std::cout << "kernel launchs: " << NSizes / sizesPerLaunch  << std::endl;

	int* d_configs = CudaGenerate::createDeviceRoomConfigsArray(rooms);
	int* d_perm = CudaGenerate::createDevicePermArray();
	int* d_adj = CudaGenerate::createDeviceAdjArray(rooms, allReq, allReqCount);
	int* d_adj_count = CudaGenerate::createDeviceAdjCountArray(rooms);
	int16_t* d_res = CudaGenerate::createDeviceResArray(result_mem_size);

	int qtdThreadY = sizesPerLaunch > targetQtdThreadsX ? targetQtdThreadsX : sizesPerLaunch;
	int qtdBlocksX = (sizesPerLaunch + qtdThreadY - 1) / qtdThreadY;

	if(qtdThreadY < GenerateHandler::minThreadCount()){
		qtdThreadY = GenerateHandler::minThreadCount();
	}
	
	std::cout << "targetThreadsPerBlock: " << targetThreadsPerBlock << ", targetQtdThreadsX: " << targetQtdThreadsX << std::endl;
	std::cout << "qtdThreadY: " << qtdThreadY << ", qtdBlocksX: " << qtdBlocksX << std::endl;

	if(GenerateHandler::checkThreadCountValue(qtdThreadY))
		return;

	int dependencyControl = 0;
	// const int nThreads = targetRamSize / targetVRamSize;
	const int nThreads = 1;
	// int16_t** h_res = CudaGenerate::createHostResArray(result_mem_size, nThreads);
	std::vector<std::vector<int16_t>> h_res(nThreads, std::vector<int16_t>(ptsPerKernel, -1));
	// std::vector<std::vector<int16_t>> h_res(1, std::vector<int16_t>(ptsPerKernel, -1));

	std::cout << "nThreads: " << nThreads << std::endl;

    #pragma omp parallel num_threads(nThreads)
    {
        #pragma omp single
        {
			for(int i = 0; i < NSizes; i+= sizesPerLaunch){
                #pragma omp task depend(inout: dependencyControl) priority(0)
                {
					int diff = NSizes - i;
					int threadId = omp_get_thread_num();
					// int threadId = 0;
					dependencyControl = i / sizesPerLaunch;

					printf("producer %d init, diff: %d\n", threadId, diff);
					if(diff < sizesPerLaunch){
						int final_qtdBlocksX = (diff + qtdThreadY - 1) / qtdThreadY;
						CudaGenerate::launchGenereteKernel(final_qtdBlocksX, qtdThreadY, NConn, NPerm, diff, d_configs, d_perm, d_adj, d_adj_count, d_res, h_res[threadId].data(), i, result_mem_size);
					} else {
						CudaGenerate::launchGenereteKernel(qtdBlocksX, qtdThreadY, NConn, NPerm, sizesPerLaunch, d_configs, d_perm, d_adj, d_adj_count, d_res, h_res[threadId].data(), i, result_mem_size);
					}

    				// std::vector<int16_t> result_copy(h_res[threadId]); 
					CudaGenerate::launchDuplicateCheckKernel(d_res, h_res[threadId].data(), layoutsPerKernel, result_mem_size);
					
					printf("producer %d end\n", threadId);

                	#pragma omp task priority(10)
					{
						// printf("consumer %d end (pts: %zu, layouts: %zu)\n", threadId, h_res[threadId].size(), h_res[threadId].size() / __GENERATE_RES_LENGHT);
						GenerateHandler::consume(h_res[threadId], hdd, combid, dependencyControl);
					}
                }
            }
        }
    }
	printf("parallel end\n");

    CudaGenerate::freeDeviceArrays(d_configs, d_perm, d_adj, d_adj_count, d_res);
	// CudaGenerate::freeHostArrays(h_res, nThreads);
}

int GenerateHandler::minThreadCount(){
	if((__GENERATE_N * __ROOM_CONFIG_LENGHT) < __SIZE_ADJ)
		return __SIZE_ADJ;

	return __GENERATE_N * __ROOM_CONFIG_LENGHT;
}

int GenerateHandler::checkThreadCountValue(const int qtdThreadsY){
	if(qtdThreadsY < __GENERATE_N * __ROOM_CONFIG_LENGHT || qtdThreadsY < __SIZE_ADJ || qtdThreadsY < __SIZE_ADJ_TYPES){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!! Not enough threads !!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	return 0;
}

int GenerateHandler::checkDefineValues(const std::vector<RoomConfig>& rooms, const int reqSize){
	const int n = rooms.size();

	if(n != __GENERATE_N){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!! wrong __GENERATE_N value !!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	if(n * n != __GENERATE_REQ_ADJ){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!! wrong __GENERATE_REQ_ADJ value !!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	if(reqSize * reqSize != __SIZE_ADJ){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!! wrong __SIZE_ADJ value !!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	if(reqSize != __SIZE_ADJ_TYPES){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!! wrong __SIZE_ADJ_TYPES value !!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}	

	long perms = Calculator::Factorial(n);
	if(perms != __GENERATE_PERM){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!! wrong __GENERATE_PERM value !!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	long rotations = Calculator::NRotations(n);
	if(rotations != __GENERATE_ROTATIONS){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!! wrong __GENERATE_ROTATIONS value !!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}	

	if(((n * 4) + 1) != __GENERATE_RES_LENGHT){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!! wrong __GENERATE_RES_LENGHT value !!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	if((n * 2) != __GENERATE_SIZE_LENGHT){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!! wrong __GENERATE_SIZE_LENGHT value !!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	if((n * 4) != __GENERATE_RES_LAYOUT_LENGHT){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!! wrong __GENERATE_RES_LAYOUT_LENGHT value !!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	return 0;
}