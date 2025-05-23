#include <iostream>
#include <algorithm>
#include <omp.h>
#include <chrono>

#include "combineHandler.h"
#include "combine.cuh"
#include "../lib/calculator.h"
#include "../lib/log.h"
#include "../lib/storage.h"
#include "../lib/globals.h"
// #include "../lib/cvHelper.h"


CombineHandler::CombineHandler(){}


// TODO split consume with more threads, too many cores doing nothing
void CombineHandler::consume(const std::vector<int>& h_res, const size_t res_mem_size, Storage& hdd, const int combId, const int combFilesdId, const int taskCount, const int max_layout_size){
	// const int threadId = omp_get_thread_num();
	const int max_size_diff = max_layout_size;

	// if(combId != 2293788)
	// 	return;

    // const size_t invalid_idx = (size_t)-1;
	const size_t res_size = res_mem_size / sizeof(int);

	// std::vector<int> result; //sorted result
	// std::vector<size_t> h_begin(max_size_diff, invalid_idx);
	std::vector<std::vector<int>> fileMaxH(max_size_diff, std::vector<int>(max_size_diff, -1));
	std::vector<std::vector<int>> fileMaxW(max_size_diff, std::vector<int>(max_size_diff, -1));
	std::vector<std::vector<std::vector<int>>> results(max_size_diff, std::vector<std::vector<int>>(max_size_diff, std::vector<int>()));

	// std::cout << "combine init" << std::endl;
	// size_t lowest_index_table_idx = max_size_diff * max_size_diff;

	for(size_t i = 0; i < res_size; i+= __SIZE_RES){
		if(h_res[i] == __COMBINE_INVALID_LAYOUT)
			continue;
			
		int min_diff_H = h_res[i + __COMBINE_RES_DIFF_H];
		int min_diff_W = h_res[i + __COMBINE_RES_DIFF_W];
		int max_diff_H = h_res[i + __COMBINE_RES_DIFF_H];
		int max_diff_W = h_res[i + __COMBINE_RES_DIFF_W];
		const int area = h_res[i + __COMBINE_RES_AREA];
		const int a_layout_idx = h_res[i + __COMBINE_RES_A_IDX];
		const int b_layout_idx = h_res[i + __COMBINE_RES_B_IDX];

		for(; i < res_size; i+= __SIZE_RES){
			if(h_res[i] == __COMBINE_INVALID_LAYOUT)
				continue;

			if(h_res[i + __COMBINE_RES_A_IDX] != a_layout_idx || h_res[i + __COMBINE_RES_B_IDX] != b_layout_idx)
				break;

			const int diff_H = h_res[i + __COMBINE_RES_DIFF_H];
			const int diff_W = h_res[i + __COMBINE_RES_DIFF_W];
			if(min_diff_H > diff_H)
				min_diff_H = diff_H;
			
			if(min_diff_W > diff_W)
				min_diff_W = diff_W;

			if(max_diff_H < diff_H)
				max_diff_H = diff_H;
			
			if(max_diff_W < diff_W)
				max_diff_W = diff_W;
		}
		i -= __SIZE_RES;


		// int minSizeId = (min_diff_H << __RES_FILE_LENGHT_BITS) | min_diff_W;
		// if(minSizeId > 300000)
		// 	continue;
		// if(min_diff_H != 45 || min_diff_W != 95)

		// if(min_diff_H == 40 && min_diff_W == 95 && taskCount == 0){
		// 	int error = (i / __SIZE_RES) % 8;
		// 	if(error % 2 == 0) error = -error;
		// 	max_diff_H += error;
		// 	max_diff_W += error;
		// }

		if(fileMaxH[min_diff_H][min_diff_W] < max_diff_H)
			fileMaxH[min_diff_H][min_diff_W] = max_diff_H;

		if(fileMaxW[min_diff_H][min_diff_W] < max_diff_W)
			fileMaxW[min_diff_H][min_diff_W] = max_diff_W;

		std::vector<int>& arr = results[min_diff_H][min_diff_W];

		// const int minSizeId = (min_diff_H << __RES_FILE_LENGHT_BITS) | min_diff_W;
		// if(combId == 917553 && minSizeId == 163900){
		// 	std::cout << std::endl << std::endl << std::endl << taskCount << ", max_diff_H: " << max_diff_H << ", max_diff_W: " << max_diff_W << ", " << std::endl;

		// 	for(size_t j = 0; j < arr.size(); j+=__SIZE_RES_DISK){
		// 		for(size_t k = 0; k < __SIZE_RES_DISK; k++){
		// 			std::cout << arr[j + k] << ", ";
		// 		}
		// 		std::cout << std::endl;
		// 	}
		// }

		// TODO replace with binary search
		size_t insertIdx = 0;
		while(insertIdx < arr.size() && 
			 (arr[insertIdx + __RES_DISK_MAX_H] < max_diff_H || 
			 (arr[insertIdx + __RES_DISK_MAX_H] == max_diff_H && arr[insertIdx + __RES_DISK_MAX_W] < max_diff_W))){
			insertIdx += __SIZE_RES_DISK;
		}
		while(insertIdx < arr.size() && arr[insertIdx + __RES_DISK_MAX_H] == max_diff_H && arr[insertIdx + __RES_DISK_MAX_W] == max_diff_W){
			insertIdx += __SIZE_RES_DISK;
		}

		// if(combId == 917553 && minSizeId == 163900){
		// 	std::cout << ", insertIdx: " << insertIdx << ", prev values: " << 
		// }


		size_t original_arr_size = arr.size();
		arr.resize(arr.size() + __SIZE_RES_DISK);

		if(insertIdx < original_arr_size){
			std::shift_right(begin(arr) + insertIdx, end(arr), __SIZE_RES_DISK);
		}

		arr[insertIdx + __RES_DISK_MAX_H] = max_diff_H;
		arr[insertIdx + __RES_DISK_MAX_W] = max_diff_W;
		arr[insertIdx + __RES_DISK_A_IDX] = a_layout_idx;
		arr[insertIdx + __RES_DISK_B_IDX] = b_layout_idx;
		arr[insertIdx + __RES_DISK_AREA] = area;

		// if(combId == 917553 && minSizeId == 163900){
		// 	std::cout << std::endl << "insertIdx: " << insertIdx << ", " << std::endl;

		// 	for(size_t j = 0; j < arr.size(); j+=__SIZE_RES_DISK){
		// 		for(size_t k = 0; k < __SIZE_RES_DISK; k++){
		// 			std::cout << arr[j + k] << ", ";
		// 		}
		// 		std::cout << std::endl;
		// 	}

		// 	getchar();
		// }

		// if(min_diff_H == 40 && min_diff_W == 95){
		// 	std::cout << "min H: " << min_diff_H << ", min W: " << min_diff_W << ", max H: " << max_diff_H << ", max W: " << max_diff_W << ", a idx: " << a_layout_idx << ", b idx: " << b_layout_idx << ", insertIdx: " << insertIdx  << std::endl;
		// 	for(int j = 0; j < arr.size(); j+=__SIZE_RES_DISK){
		// 		for(int k = 0; k < __SIZE_RES_DISK; k++){
		// 			std::cout << arr[j + k] << ", ";
		// 		}
		// 		std::cout << std::endl;
		// 	}
		// 	std::cout << std::endl << std::endl << std::endl;
		// 	getchar();
		// }
    }

	uint64_t totalSize = 0;
    for(int i = 0; i < max_size_diff; i++){
        for(int j = 0; j < max_size_diff; j++){
			// if(results[i][j].size() > 0){
			// 	int minSizeId = (i << __RES_FILE_LENGHT_BITS) | j;
			// 	std::cout << "consume combId: " << combId << ", minSizeId: " << minSizeId << ", layouts: " << results[i][j].size() / __SIZE_RES_DISK << ", size: " << (((double)(results[i][j].size() * sizeof(int))) / 1024.0 / 1024.0) << " MB" << std::endl;  
			// }
			totalSize += results[i][j].size();
        }
    }

    // for(int i = 0; i < results[40][95].size(); i+=__SIZE_RES_DISK){
    //     for(int j = 0; j < __SIZE_RES_DISK; j++){
	// 		std::cout << results[40][95][i + j] << ", ";
	// 	}
	// 	std::cout << std::endl;
	// }
	// std::cout << std::endl << std::endl << std::endl;

	// std::cout << "fileMaxH" << std::endl;
	// Log::printVector1D(fileMaxH);
	// std::cout << "fileMaxW" << std::endl;
	// Log::printVector1D(fileMaxW);


	hdd.saveCombineResultPart(results, combId, combFilesdId, taskCount, fileMaxH, fileMaxW);
	// std::cout << "consumer : " << threadId << " end, layouts: " << totalSize / __SIZE_RES_DISK << ", mem size (MB): " << ((double)(totalSize * sizeof(size_t))) / 1024.0 / 1024.0 << std::endl;
	std::cout << "consumer end: " << combId << ", " << ", " << taskCount << ", layouts: " << totalSize / __SIZE_RES_DISK << ", mem size (MB): " << ((double)(totalSize * sizeof(size_t))) / 1024.0 / 1024.0 << std::endl;
}

// TODO removed repeated connections (create two conn arrays, one for the a, other for the b, maybe?)
std::vector<int> CombineHandler::createConns(
	const int rooms_count_a, 
	const int rooms_count_b)
{
	std::vector<int> conns;

	const int pts_a = rooms_count_a * 4;
	const int pts_b = rooms_count_b * 4;

	for(int i = 0; i < pts_a; i++){
		const int src_offset = (i / 4) * 4;
		const int src_conn = i % 4;

		int src_x = (src_conn == 0 || src_conn == 2) ? src_offset + 0 : src_offset + 2;
		int src_y = (src_conn == 0 || src_conn == 1) ? src_offset + 1 : src_offset + 3;

		for(int j = 0; j < pts_b; j++){
			const int dst_offset = (j / 4) * 4;
			const int dst_conn = j % 4;

			if(dst_conn == src_conn)
				continue;

			int dst_x = (dst_conn == 0 || dst_conn == 2) ? dst_offset + 0 : dst_offset + 2;
			int dst_y = (dst_conn == 0 || dst_conn == 1) ? dst_offset + 1 : dst_offset + 3;

			int conn = 0;
			conn |= src_x << __COMBINE_CONN_SRC_X_SHIFT;
			conn |= src_y << __COMBINE_CONN_SRC_Y_SHIFT;
			conn |= dst_x << __COMBINE_CONN_DST_X_SHIFT;
			conn |= dst_y << __COMBINE_CONN_DST_Y_SHIFT;

			conns.push_back(conn);			
		}
	}

	// for(int conn : conns){
	// 	std::cout << conn << "\tsrc: " << ((conn >> __COMBINE_CONN_SRC_X_SHIFT) & __COMBINE_CONN_BITS) << ", " << ((conn >> __COMBINE_CONN_SRC_Y_SHIFT) & __COMBINE_CONN_BITS) << "\tdst: " << ((conn >> __COMBINE_CONN_DST_X_SHIFT) & __COMBINE_CONN_BITS) << ", " << ((conn >> __COMBINE_CONN_DST_Y_SHIFT) & __COMBINE_CONN_BITS) << std::endl;
	// }

	return conns;
}

int CombineHandler::getMaxLayoutSize(const std::vector<RoomConfig>& rooms_a, const std::vector<RoomConfig>& rooms_b){
	int max_layout_size = 0;

	for(RoomConfig room : rooms_a){
		if(room.maxH > room.maxW)
			max_layout_size += room.maxH;
		else
			max_layout_size += room.maxW;
	}

	for(RoomConfig room : rooms_b){
		if(room.maxH > room.maxW)
			max_layout_size += room.maxH;
		else
			max_layout_size += room.maxW;
	}

	return max_layout_size;
}

// int CombineHandler::getMinLayoutSize(const std::vector<RoomConfig>& rooms_a, const std::vector<RoomConfig>& rooms_b){
// 	int min_layout_size = 10000;

// 	for(RoomConfig room : rooms_a){
// 		if(room.minH < min_layout_size)
// 			min_layout_size = room.minH;

// 		if(room.minW < min_layout_size)
// 			min_layout_size = room.minW;
// 	}

// 	for(RoomConfig room : rooms_b){
// 		if(room.minH < min_layout_size)
// 			min_layout_size = room.minH;

// 		if(room.minW < min_layout_size)
// 			min_layout_size = room.minW;
// 	}

// 	return min_layout_size;
// }

int CombineHandler::getRoomsCombId(const std::vector<RoomConfig>& rooms){
	int combId = 0;
	for(RoomConfig room : rooms){
		combId += room.id;
	}

	return combId;
}

void CombineHandler::combine(
	const std::vector<RoomConfig>& rooms_a, 
	const std::vector<RoomConfig>& rooms_b, 
	const std::vector<int16_t>& a, 
	const std::vector<int16_t>& b,
	const int combFilesdId,
	std::vector<int> allReqAdj, 
	Storage& hdd)
{
	std::vector<int> conns = CombineHandler::createConns(rooms_a.size(), rooms_b.size());

	if(CombineHandler::checkDefineValues(rooms_a, rooms_b, allReqAdj))
		return;

	int combId = CombineHandler::getRoomsCombId(rooms_a) << __COMBINE_NAME_ROOMS_ID_SHIFT;
	combId |= CombineHandler::getRoomsCombId(rooms_b);

	// if(combId != 917553)
	// 	return;

	// const size_t targetRamSize = 25l * 1024l * 1024l * 1024l;
	// const size_t targetVRamSize = 8l * 1024l * 1024l * 1024l;
	const size_t targetVRamSize = 7500l * 1024l * 1024l;
	// const size_t targetVRamSize = 4000l * 1024l * 1024l;

	const int NConn = conns.size();
	const int qtd_a = a.size() / __SIZE_A_DISK;
	const int qtd_b = b.size() / __SIZE_B_DISK;
	
	const long maxResCount = targetVRamSize / (__SIZE_RES * sizeof(int));
	const long maxQtd_a = maxResCount / (qtd_b * NConn);
	const int num_a = qtd_a > maxQtd_a ? maxQtd_a : qtd_a;

	const long qtd_res = num_a * NConn * qtd_b;
	const long ptsPerKernel = qtd_res * __SIZE_RES;

	const int max_layout_size = CombineHandler::getMaxLayoutSize(rooms_a, rooms_b);
	// const int min_layout_size = CombineHandler::getMinLayoutSize(rooms_a, rooms_b);

	std::cout << std::endl;
	std::cout << "a.size(): " << a.size() << ", b.size(): " << b.size() << std::endl;
	std::cout << "qtd_a: " << qtd_a << ", qtd_b: " << qtd_b  << ", a*b: " << ((unsigned long)qtd_a) * ((unsigned long)qtd_b) << std::endl;
	std::cout << "targetVRamSize: " << targetVRamSize << ", targetVRamSize GB: " << (double)targetVRamSize / 1024.0 / 1024.0 / 1024.0 << std::endl;
	std::cout << "maxResCount: " << maxResCount << ", qtd_res: " << qtd_res << std::endl;
	std::cout << "num_a: " << num_a << ", kernel launchs: " << ((qtd_a + num_a - 1) / (num_a)) << std::endl;
	
	const long resLayoutSize = sizeof(int) * __SIZE_RES;
	const unsigned long mem_size_res = resLayoutSize * qtd_res;

	int* d_adj = CudaCombine::createDeviceAdjArray(allReqAdj);
	int* d_conns = CudaCombine::createDeviceConnArray(conns);
	int* d_res = CudaCombine::createDeviceResArray(mem_size_res);
	int16_t* d_a = CudaCombine::createDeviceCoreLayoutsArray(a);
	int16_t* d_b = CudaCombine::createDeviceCoreLayoutsArray(b);

	const int nCpuThreads = 3;
	// int16_t** h_res = CudaGenerate::createHostResArray(result_mem_size, nCpuThreads);
	std::vector<std::vector<int>> h_res(nCpuThreads, std::vector<int>(ptsPerKernel, __COMBINE_INVALID_LAYOUT));
	// std::cout << "nCpuThreads: " << nCpuThreads << std::endl;

	
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
    std::chrono::time_point<std::chrono::high_resolution_clock> clock_begin, clock_now, clock_end;
    clock_begin = std::chrono::high_resolution_clock::now();

	std::cout << "start parallel process" << std::endl;
    #pragma omp parallel num_threads(nCpuThreads)
    {
        #pragma omp single
        {
			for(int i = 0; i < qtd_a; i += num_a){
                #pragma omp task depend(inout: dependencyControl) priority(0)
                {
					// int i = qtd_a - 417490;
					int diff = qtd_a - i;
					int threadId = omp_get_thread_num();
					clock_now = std::chrono::high_resolution_clock::now();
    				std::chrono::milliseconds milis = std::chrono::duration_cast<std::chrono::milliseconds>(clock_now - clock_begin);

					dependencyControl++;
					double pctCompletion = (double)i / (double)qtd_a;
					double etf = ((double)(diff * milis.count())) / ((double)i) / 60000.0;
					double elapsed = ((double)milis.count()) / 60000.0;
					// std::chrono::minutes elapsed_minutes = std::chrono::duration_cast<std::chrono::minutes>(milis);
					// std::chrono::hours elapsed_hours = std::chrono::duration_cast<std::chrono::hours>(milis);

					// printf("producer %d init, diff: %d, completion %.4lf, etf: %.2lf minutes\n", threadId, diff, pctCompletion, etf);
					std::cout << "producer " << threadId << " init, diff: " << diff << ", completion " << pctCompletion << std::endl;
					std::cout << "elapsed: " << elapsed << " minutes (" << (elapsed / 60.0) << ") hours" << std::endl;
					std::cout << "etf: " << etf << " minutes (" << (etf / 60.0) << ") hours" << std::endl;

					if(diff < num_a){
						int final_qtdBlocksX = (diff + qtdThreadX - 1) / qtdThreadX;
						CudaCombine::createPts(mem_size_res, NConn, diff, qtd_b, i, final_qtdBlocksX, qtdThreadX, h_res[threadId].data(), d_adj, d_res, d_conns, d_a, d_b);
					} else {
						CudaCombine::createPts(mem_size_res, NConn, num_a, qtd_b, i, num_blocks, qtdThreadX, h_res[threadId].data(), d_adj, d_res, d_conns, d_a, d_b);
					}

					// CudaGenerate::launchDuplicateCheckKernel(d_res, h_res[threadId].data(), layoutsPerKernel, result_mem_size);
					
					printf("producer %d end\n", threadId);

                	#pragma omp task priority(10)
					{
						CombineHandler::consume(h_res[threadId], mem_size_res, hdd, combId, combFilesdId, dependencyControl - 1, max_layout_size);
						// CombineHandler::drawResult(h_res[threadId].data(), mem_size_res);
					}
                }
				// break;
            }
        }
    }
	printf("parallel end\n");

    CudaCombine::freeDeviceArrays(d_adj, d_res, d_conns, d_a, d_b);
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

	int maxAdjTypeSupported = 0;
	for(int i = 0; i < __RID_BITS_SIZE; i++){
		maxAdjTypeSupported |= 1 << i;
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

	if(maxAdjTypeSupported < __SIZE_ADJ_TYPES || maxAdjTypeSupported < maxAdjType){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!! wrong __SIZE_ADJ_TYPES value !!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	if(maxAdjTypeSupported != __RID_BITS){
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!! wrong __RID_BITS value !!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}
	int pts_a = a.size() * 4;
	if ((pts_a - 1) > __COMBINE_CONN_BITS)
	{
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!! wrong __COMBINE_CONN_BITS value !!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	int pts_b = b.size() * 4;
	if ((pts_b - 1) > __COMBINE_CONN_BITS)
	{
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!! wrong __COMBINE_CONN_BITS value !!!!!!!!!!!!!!!!!!!!!" << std::endl;
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		return 1; 
	}

	return 0;
}
