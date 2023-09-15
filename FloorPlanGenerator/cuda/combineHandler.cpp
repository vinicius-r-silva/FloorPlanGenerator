#include <iostream>
#include <algorithm>
#include <omp.h>

#include "combineHandler.h"
#include "combine.cuh"
#include "../lib/calculator.h"
#include "../lib/log.h"
#include "../lib/storage.h"
#include "../lib/globals.h"
#include "../lib/cvHelper.h"


CombineHandler::CombineHandler(){}

void CombineHandler::consume(const int *h_res, const size_t res_mem_size, Storage& hdd, const int taskCount, const int max_layout_size){
	int threadId = omp_get_thread_num();
	std::cout << "consumer " << threadId << " init, count: " << taskCount << ", max_layout_size: " << max_layout_size << ", h_res[0]: " << h_res[0] << ", result path: " << hdd.getResultPath() << ", res_mem_size: " << res_mem_size << std::endl;

	const int qtd_pts = __SIZE_RES / 2;
    std::vector<int16_t> ptsX(qtd_pts, 0); 
    std::vector<int16_t> ptsY(qtd_pts, 0);

    std::vector<int> iArray;
    for(size_t i = 0; i < res_mem_size; i += __SIZE_RES){
		if(h_res[i] == -1)
			continue;

		// i = 1660128;

        std::cout << "i: " << i << ", idx: " << i / qtd_pts << std::endl;
        for(int j = 0; j < qtd_pts; j++){
            ptsX[j] = h_res[i + (j * 2)];
            ptsY[j] = h_res[i + (j * 2) + 1];

            std::cout << "(" << ptsX[j] << ", " << ptsY[j] << "), ";
        }
        std::cout << std::endl;
        

        int dir = CVHelper::showLayoutMove(ptsX, ptsY);

        if(dir == -1 && iArray.size() == 0){
            i = -__SIZE_RES;
		} else if(dir == -1 ){
			i = iArray.back() - __SIZE_RES; iArray.pop_back(); 
        } else {
			iArray.push_back(i);
        }
    }
	// printf("consumer %d init, count: %d", threadId, taskCount);
	
	// std::vector<int> result;
	// std::vector<int> h_begin(max_layout_size, 0);
	// std::vector<int> index_table(max_layout_size * max_layout_size, 0); //relative

	// for(int i = 0; i < res_mem_size; i+= __SIZE_RES){
	// 	if(h_res[i] == 0)
	// 		continue;

    //     // if(h_res[i] != 40 || h_res[i + 2] != 0)
    //     //     continue;

	// 	const int diffH = h_res[i];
	// 	const int diffW = h_res[i + 1];
	// 	const int a_layout_idx = h_res[i + 2];
	// 	const int b_layout_idx = h_res[i + 3];
		
	// 	const int h_table_start = diffH * max_layout_size;
	// 	const int h_table_end = h_table_start + max_layout_size;
	// 	const int h_table_idx = h_table_start + diffW;
	// 	const int insert_idx = h_begin[diffH] + index_table[h_table_idx];

	// 	for(int j = diffH + 1; j < max_layout_size; j++){
	// 		h_begin[j] += __SIZE_RES;
	// 	}
	// 	for(int j = h_table_idx; j < h_table_end; j++){
	// 		index_table[j] += __SIZE_RES;
	// 	}

	// 	result.resize(result.size() + __SIZE_RES);
	// 	std::shift_right(result.begin() + insert_idx, result.end(), __SIZE_RES);
	// 	result[insert_idx] = diffH;
	// 	result[insert_idx + 1] = diffW;
	// 	result[insert_idx + 2] = a_layout_idx;
	// 	result[insert_idx + 3] = b_layout_idx;

    //     // std::cout << "diffH: " << diffH << ", diffW: " << diffW << ", a_layout_idx: " << a_layout_idx << ", b_layout_idx: " << b_layout_idx << ", insert_idx: " << insert_idx << std::endl;
	// }

	// // const int max_layout_size = 200;
	// // std::vector<int> result;
	// // std::vector<int> h_begin(max_layout_size, 0);
	// // std::vector<int> index_table(max_layout_size * max_layout_size, 0); //relative

	// // hdd.saveResult(result, combid, taskCount);
	// // printf("consumer %d end (pts: %zu, layouts: %zu)\n", threadId, result.size(), result.size() / __GENERATE_RES_LENGHT);
}

void CombineHandler::combine(
	const std::vector<RoomConfig>& rooms_a, 
	const std::vector<RoomConfig>& rooms_b, 
	const std::vector<int16_t>& a, 
	const std::vector<int16_t>& b,
	std::vector<int> allReqAdj, 
	Storage& hdd)
{

	if(CombineHandler::checkDefineValues(rooms_a, rooms_b, allReqAdj))
		return;

	// const size_t targetRamSize = 25l * 1024l * 1024l * 1024l;
	const size_t targetVRamSize = 8l * 1024l * 1024l * 1024l;

	const int NConn = __COMBINE_CONN;
	const int qtd_a = a.size() / __SIZE_A_DISK;
	const int qtd_b = b.size() / __SIZE_B_DISK;
	
	const long maxResCount = targetVRamSize / (__SIZE_RES * sizeof(int));
	const long maxQtd_a = maxResCount / (qtd_b * NConn);
	const int num_a = qtd_a > maxQtd_a ? maxQtd_a : qtd_a;

	const long qtd_res = num_a * NConn * qtd_b;
	const long ptsPerKernel = qtd_res * __SIZE_RES;

	std::cout << std::endl;
	std::cout << "a.size(): " << a.size() << ", b.size(): " << b.size() << std::endl;
	std::cout << "qtd_a: " << qtd_a << ", qtd_b: " << qtd_b  << ", a*b: " << qtd_a * qtd_b << std::endl;
	std::cout << "maxResCount: " << maxResCount << ", qtd_res: " << qtd_res << std::endl;
	std::cout << "num_a: " << num_a << ", kernel launchs: " << ((qtd_a + num_a - 1) / (num_a)) << std::endl;

	const long resLayoutSize = sizeof(int) * __SIZE_RES;
	const unsigned long mem_size_res = resLayoutSize * qtd_res;

	int* d_adj = CudaCombine::createDeviceAdjArray(allReqAdj);
	int* d_res = CudaCombine::createDeviceResArray(mem_size_res);
	int16_t* d_a = CudaCombine::createDeviceCoreLayoutsArray(a);
	int16_t* d_b = CudaCombine::createDeviceCoreLayoutsArray(b);

	const int nCpuThreads = 1;
	// int16_t** h_res = CudaGenerate::createHostResArray(result_mem_size, nCpuThreads);
	std::vector<std::vector<int>> h_res(nCpuThreads, std::vector<int>(ptsPerKernel, -1));
	std::cout << "nCpuThreads: " << nCpuThreads << std::endl;

	
	int qtdThreadX = qtd_b > __THREADS_PER_BLOCK ? __THREADS_PER_BLOCK : qtd_b; 
	int num_blocks = (qtd_b + qtdThreadX -1) / qtdThreadX;

	if(qtdThreadX < CombineHandler::minThreadCount()){
		qtdThreadX = CombineHandler::minThreadCount();
	}

	if(CombineHandler::checkThreadCountValue(qtdThreadX))
		return;

	std::cout << "num_blocks: " << num_blocks << std::endl;
	std::cout << "qtdThreadX: " << qtdThreadX << std::endl;

	int dependencyControl = 0;



    // #pragma omp parallel num_threads(nThreads)
    // {
    //     #pragma omp single
    //     {
				for(int i = 0; i < qtd_a; i += num_a){
                // #pragma omp task depend(inout: dependencyControl) priority(0)
                // {
					int diff = qtd_a - i;
					int threadId = omp_get_thread_num();
					dependencyControl++;

					// std::cout << ((float)i / (float)qtd_a) * 100.0 <<  " %" << std::endl;
					printf("producer %d init, diff: %d\n", threadId, diff);
					if(diff < num_a){
						int final_qtdBlocksX = (diff + qtdThreadX - 1) / qtdThreadX;
						CudaCombine::createPts(mem_size_res, NConn, diff, qtd_b, i, final_qtdBlocksX, qtdThreadX, h_res[threadId].data(), d_adj, d_res, d_a, d_b);
					} else {
						CudaCombine::createPts(mem_size_res, NConn, num_a, qtd_b, i, num_blocks, qtdThreadX, h_res[threadId].data(), d_adj, d_res, d_a, d_b);
					}

					// CudaGenerate::launchDuplicateCheckKernel(d_res, h_res[threadId].data(), layoutsPerKernel, result_mem_size);
					
					printf("producer %d end\n", threadId);

                	// #pragma omp task priority(10)
					// {
						CombineHandler::consume(h_res[threadId].data(), mem_size_res, hdd, dependencyControl - 1, 200);
					// }
                // }
            }
        // }
    // }
	printf("parallel end\n");

    CudaCombine::freeDeviceArrays(d_adj, d_res, d_a, d_b);
}

int CombineHandler::minThreadCount(){
	if(__SIZE_A_DISK < __SIZE_ADJ)
		return __SIZE_ADJ;

	return __SIZE_A_DISK;
}

int CombineHandler::checkThreadCountValue(const int qtdThreadsY){
	if(qtdThreadsY < __SIZE_A_DISK || qtdThreadsY < __SIZE_ADJ){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!! Not enough threads !!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	return 0;
}

int CombineHandler::checkDefineValues(const std::vector<RoomConfig>& a, const std::vector<RoomConfig>& b, std::vector<int> adj){
	if(a.size() != __COMBINE_N_A){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!! wrong __COMBINE_N_A value !!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	if(b.size() != __COMBINE_N_B){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!! wrong __COMBINE_N_B value !!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	if(adj.size() != __SIZE_ADJ){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!! wrong __SIZE_ADJ value !!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	if(adj.size() != __SIZE_ADJ_TYPES * __SIZE_ADJ_TYPES){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!! wrong __SIZE_ADJ_TYPES value !!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	if(a.size() * 4 != __SIZE_A_LAYOUT){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!! wrong __SIZE_A_LAYOUT value !!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	if(b.size() * 4 != __SIZE_B_LAYOUT){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!! wrong __SIZE_B_LAYOUT value !!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	if((a.size() * 4) + 1 != __SIZE_A_DISK){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!! wrong __SIZE_A_DISK value !!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	if((b.size() * 4) + 1 != __SIZE_B_DISK){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!! wrong __SIZE_B_DISK value !!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	if(b.size() + a.size() - 1 != __CONN_CHECK_IDX){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!! wrong __CONN_CHECK_IDX value !!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	int connCheck = 0;
	for(size_t i = 0; i < b.size() + a.size(); i++){
		connCheck |= 1 << i;
	}

	if(connCheck != __CONN_CHECK){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!! wrong __CONN_CHECK value !!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	int maxAdjTypes = 0;
	for(int i = 0; i < __RID_BITS_SIZE; i++){
		maxAdjTypes |= 1 << i;
	}

	int maxAdjType = 0;
	for(RoomConfig room : a){
		if(maxAdjType < room.rPlannyId)
			maxAdjType = room.rPlannyId;
	}
	for(RoomConfig room : b){
		if(maxAdjType < room.rPlannyId)
			maxAdjType = room.rPlannyId;
	}

	if(maxAdjTypes < __SIZE_ADJ_TYPES || maxAdjTypes < maxAdjType){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!! wrong __SIZE_ADJ_TYPES value !!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	if(maxAdjTypes != __RID_BITS){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!! wrong __RID_BITS value !!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	return 0;
}