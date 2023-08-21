#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <filesystem>

#include "process.h"
#include "../lib/globals.h"

// void CudaProcess::processResult(std::vector<int>& result, const int *h_res, const int res_size){
void CudaProcess::processResult(std::vector<int>& result, const int *h_res, const int res_size, std::vector<int>& h_begin, std::vector<int>& index_table, const int max_layout_size){
	for(int i = 0; i < res_size; i+= __SIZE_RES){
		if(h_res[i] == 0)
			continue;

        // if(h_res[i] != 40 || h_res[i + 2] != 0)
        //     continue;

		const int diffH = h_res[i];
		const int diffW = h_res[i + 1];
		const int a_layout_idx = h_res[i + 2];
		const int b_layout_idx = h_res[i + 3];
		
		const int h_table_start = diffH * max_layout_size;
		const int h_table_end = h_table_start + max_layout_size;
		const int h_table_idx = h_table_start + diffW;
		const int insert_idx = h_begin[diffH] + index_table[h_table_idx];

		for(int j = diffH + 1; j < max_layout_size; j++){
			h_begin[j] += __SIZE_RES;
		}
		for(int j = h_table_idx; j < h_table_end; j++){
			index_table[j] += __SIZE_RES;
		}

		result.resize(result.size() + __SIZE_RES);
		std::shift_right(result.begin() + insert_idx, result.end(), __SIZE_RES);
		result[insert_idx] = diffH;
		result[insert_idx + 1] = diffW;
		result[insert_idx + 2] = a_layout_idx;
		result[insert_idx + 3] = b_layout_idx;

        // std::cout << "diffH: " << diffH << ", diffW: " << diffW << ", a_layout_idx: " << a_layout_idx << ", b_layout_idx: " << b_layout_idx << ", insert_idx: " << insert_idx << std::endl;
	}
}