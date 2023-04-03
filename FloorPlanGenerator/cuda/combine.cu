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
#define __N_PTS 6

#define __SIZE_A 12		// n_a * 4
#define __SIZE_B 12		// n_b * 4
#define __SIZE_PTS 24	// n_pts * 4
#define __SIZE_RES 5	// score, maxH, minH, maxW, minW //Add Area later
// #define __SIZE_NBR 9	// __SIZE_A * __SIZE_B

#define __LEFT 0
#define __UP 1
#define __RIGHT 2
#define __DOWN 3

// __global__
// void k_checkOverlap(int16_t *d_pts, int16_t *d_res, const uint max_idx){
// 	const int idx = threadIdx.x;
// 	if(idx >= max_idx)
// 		return;

// 	const int pts_idx = idx * __SIZE_PTS;
// 	const int res_idx = idx * __SIZE_RES;

// 	int16_t pts[__SIZE_PTS];
// 	for(int i = 0; i < __SIZE_PTS; i++){
// 		pts[i] = d_pts[pts_idx + i];
// 	}

// 	//left, up, right, down
// 	int16_t notOverlap = 1;
// 	for(int i = 0; i < __SIZE_A && notOverlap; i+=4){
// 		for(int j = __SIZE_A; (j < (__SIZE_A +__SIZE_B)) && notOverlap; j+=4){
// 			// if(idx > 27){
// 			// 	printf("idx: %d, i: %d, j: %d,\ta_down: %d, a_up: %d, a_left: %d, a_right: %d, b_down: %d, b_up: %d, b_left: %d, b_right: %d\n",
// 			// 	idx, i, j,
// 			// 	pts[i + __DOWN], pts[i + __UP], pts[i], pts[i + __RIGHT],
// 			// 	pts[j + __DOWN], pts[j + __UP], pts[j], pts[j + __RIGHT]);
// 			// }

// 			if(((pts[i + __DOWN] > pts[j + __UP] && pts[i + __DOWN] <= pts[j + __DOWN]) ||
// 				(pts[i + __UP]  >= pts[j + __UP] && pts[i + __UP] < pts[j + __DOWN])) &&
// 				((pts[i + __RIGHT] > pts[j] && pts[i + __RIGHT] <= pts[j + __RIGHT]) ||
// 				(pts[i]  >= pts[j] && pts[i]  <  pts[j + __RIGHT]) ||
// 				(pts[i]  <= pts[j] && pts[i + __RIGHT] >= pts[j + __RIGHT]))){
// 					notOverlap = 0;
// 			}

			
// 			else if(((pts[j + __DOWN] > pts[i + __UP] && pts[j + __DOWN] <= pts[i + __DOWN]) ||
// 				(pts[j + __UP] >= pts[i + __UP] && pts[j + __UP] < pts[i + __DOWN])) &&
// 				((pts[j + __RIGHT] > pts[i] && pts[j + __RIGHT] <= pts[i + __RIGHT]) ||
// 				(pts[j]  >= pts[i] && pts[j]  <  pts[i + __RIGHT]) ||
// 				(pts[j]  <= pts[i] && pts[j + __RIGHT] >= pts[i + __RIGHT]))){
// 					notOverlap = 0;
// 			}

			
// 			else if(((pts[i + __RIGHT] > pts[j] && pts[i + __RIGHT] <= pts[j + __RIGHT]) ||
// 				(pts[i] >= pts[j] && pts[i] < pts[j + __RIGHT])) &&
// 				((pts[i + __DOWN] > pts[j + __UP] && pts[i + __DOWN] <= pts[j + __DOWN]) ||
// 				(pts[i + __UP]  >= pts[j + __UP] && pts[i + __UP]   <  pts[j + __DOWN]) ||
// 				(pts[i + __UP]  <= pts[j + __UP] && pts[i + __DOWN] >= pts[j + __DOWN]))){
// 					notOverlap = 0;
// 			}

			
// 			else if(((pts[j + __RIGHT] > pts[i] && pts[j + __RIGHT] <= pts[i + __RIGHT]) ||
// 				(pts[j] >= pts[i] && pts[j] < pts[i + __RIGHT])) &&
// 				((pts[j + __DOWN] > pts[i + __UP] && pts[j + __DOWN] <= pts[i + __DOWN]) ||
// 				(pts[j + __UP]  >= pts[i + __UP] && pts[j + __UP]   <  pts[i + __DOWN]) ||
// 				(pts[j + __UP]  <= pts[i + __UP] && pts[j + __DOWN] >= pts[i + __DOWN]))){
// 					notOverlap = 0;
// 			}
// 		}
// 	}


// 	// if(idx > 27){
// 	// 	printf("idx: %d\t notOverlap: %d\n", idx, notOverlap);
// 	// }
// 	d_res[res_idx] = notOverlap - 1;
// }

// const int num_threads = 768; // 1024
// const int num_blocks = (qtd_b + num_threads -1) / num_threads;
// dim3 grid(num_blocks, num_a, NConn);
// dim3 threads(num_threads, 1, 1);

__global__ 
void k_createPts(int16_t *d_a, int16_t *d_b, int16_t *d_res, const int qtd_a, const int qtd_b, const int a_offset) {
	// Block and thread indexes
	// Each blockIdx.x iterates over a fixed number (num_a) of A layouts (blockIdx.y), that iterates over Nconn connections (blockIdx.z)
	// Each threadIdx.x represents a Layout B design inside the blockIdx.x block 

	//K represents the connection (from 0 to 15, skipping 0, 5, 10 and 15)
	const int k = blockIdx.z + 1 + blockIdx.z/4; 
	const int a_idx = (blockIdx.y + a_offset) * __SIZE_A; //layout A index
	const int b_idx = blockIdx.x * blockDim.x + threadIdx.x; //layout B index (without * __SIZE_B)
	const int res_idx = ((blockIdx.z * qtd_a + blockIdx.y) * qtd_b +  b_idx) * __SIZE_RES;
	// const int nbr_idx = ((blockIdx.z * qtd_a + blockIdx.y) * qtd_b +  b_idx) * __SIZE_NBR;

	// Check bounds
	if(b_idx >= qtd_b || blockIdx.y >= qtd_a)
		return;

	// printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, k: %d, a_idx: %d, b_idx: %d, res_idx: %d\n",blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, k, a_idx, b_idx, res_idx);
	// printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, k, a_idx, b_idx, res_idx);

	// Load A into shared memory
	__shared__ int16_t a[__SIZE_A];
	if(threadIdx.x < __SIZE_A){
		a[threadIdx.x] = d_a[a_idx + threadIdx.x];
	}
  	__syncthreads();

	// Load B into local memory
	int16_t b[__SIZE_B];
	for(int i = 0; i < __SIZE_B; i++){
		b[i] = d_b[b_idx*__SIZE_B + i];
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
		src = a[__SIZE_A - 4];
	else 
		src = a[__SIZE_A - 2];

	//Move layout B in the X axis by diffX points
	const int diffX = src - dst;
	for(int i = 0; i < __SIZE_B; i+=2){
		b[i] += diffX;
	}

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
	int8_t notOverlap = 1;
	for(int i = 0; i < __SIZE_A && notOverlap; i+=4){
		const int a_left = a[i];
		const int a_up = a[i + __UP];
		const int a_down = a[i + __DOWN];
		const int a_right = a[i + __RIGHT];

		if(a_up < minH)
			maxH = a_up;
		if(a_down > maxH)
			maxH = a_down;
		if(a_left < minW)
			minW = a_left;
		if(a_right > maxW)
			maxW = a_right;

		for(int j = 0; j < __SIZE_B && notOverlap; j+=4){
			const int b_left = b[j];
			const int b_up = b[j + __UP];
			const int b_down = b[j + __DOWN];
			const int b_right = b[j + __RIGHT];

			if(((a_down > b_up && a_down <= b_down) ||
				(a_up  >= b_up && a_up < b_down)) &&
				((a_right > b_left && a_right <= b_right) ||
				(a_left  >= b_left && a_left  <  b_right) ||
				(a_left  <= b_left && a_right >= b_right))){
					notOverlap = 0;
			}

			
			else if(((b_down > a_up && b_down <= a_down) ||
				(b_up >= a_up && b_up < a_down)) &&
				((b_right > a_left && b_right <= a_right) ||
				(b_left  >= a_left && b_left  <  a_right) ||
				(b_left  <= a_left && b_right >= a_right))){
					notOverlap = 0;
			}

			
			else if(((a_right > b_left && a_right <= b_right) ||
				(a_left >= b_left && a_left < b_right)) &&
				((a_down > b_up && a_down <= b_down) ||
				(a_up  >= b_up && a_up   <  b_down) ||
				(a_up  <= b_up && a_down >= b_down))){
					notOverlap = 0;
			}

			
			else if(((b_right > a_left && b_right <= a_right) ||
				(b_left >= a_left && b_left < a_right)) &&
				((b_down > a_up && b_down <= a_down) ||
				(b_up  >= a_up && b_up   <  a_down) ||
				(b_up  <= a_up && b_down >= a_down))){
					notOverlap = 0;
			}

			// if(!notOverlap)
			// 	break;
		}
	}
	
	printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",notOverlap,blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, k, a_idx, b_idx, res_idx);
	if(!notOverlap)
		return;
	// d_res[res_idx] = notOverlap;

	d_res[res_idx + 1] = maxH;
	d_res[res_idx + 2] = maxW;
	d_res[res_idx + 3] = minH;
	d_res[res_idx + 4] = minW;

	// printf("a_idx: %d,\tb_idx: %d,\tres_idx: %d,\tblockIdx.x: %d,\tblockIdx.y: %d,\tblockIdx.z: %d,\tdblockDim.x: %d,\tthreadIdx.x: %d\n", a_idx, b_idx, res_idx, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x);
	// printf("a_idx: %d,\tb_idx: %d,\tres_idx: %d,\tblockIdx.x: %d,\tblockIdx.y: %d,\tblockIdx.z: %d,\tdblockDim.x: %d,\tthreadIdx.x: %d,\tdiffX: %d,\tdiffY: %d,\ta[0]: %d,\ta[1]: %d\n", a_idx, b_idx, res_idx - __SIZE_A, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, diffX, diffY, a[0], a[1]);
}

// __global__ 
// void k_hellowWorld() {
// 	printf("Hello World. \n\t blockDim.x: %d, blockDim.y: %d, blockDim.z: %d\n\t blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d\n\t threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n", blockDim.x, blockDim.y, blockDim.z, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
// }

void gpuHandler::createPts(const std::vector<int16_t>& a, const std::vector<int16_t>& b) {
	const int NConn = 12;
	const long num_a = 2;
	// const int qtd_a = a.size() / __SIZE_A;
	// const int qtd_b = b.size() / __SIZE_B;
	const int qtd_a = 2;
	const int qtd_b = 2;

	findCudaDevice();	

	const long aLayoutSize = sizeof(int16_t) * __SIZE_A;
	const long bLayoutSize = sizeof(int16_t) * __SIZE_B;
	const long resLayoutSize = sizeof(int16_t) * __SIZE_RES;
	// const long nbrLayoutSize = sizeof(int8_t) * __SIZE_NBR;
	const unsigned long mem_size_a = aLayoutSize * qtd_a;
	const unsigned long mem_size_b = bLayoutSize * qtd_b;
	const unsigned long mem_size_res = num_a * NConn * qtd_b * resLayoutSize;
	// const unsigned long mem_size_nbr = num_a * NConn * qtd_b * nbrLayoutSize;

	// allocate host memory
	int16_t *h_a = (int16_t *)(&a[0]);
	int16_t *h_b = (int16_t *)(&b[0]);
	int16_t *h_res = nullptr;

	// cudaMallocHost((void**)&h_nbr, mem_size_nbr);
	cudaMallocHost((void**)&h_res, mem_size_res);

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	// setup execution parameters
	const int num_threads = 768; // 1024
	const int num_blocks = (qtd_b + num_threads -1) / num_threads;

	dim3 grid(num_blocks, num_a, NConn);
	dim3 threads(num_threads, 1, 1);
	// dim3 grid(2, 1, NConn);
	// dim3 threads(6, 1, 1);


	// allocate device memory
	int16_t *d_a, *d_b, *d_res;
	// int8_t *d_nbr;
	checkCudaErrors(cudaMalloc((void **)&d_a, mem_size_a));
	checkCudaErrors(cudaMalloc((void **)&d_b, mem_size_b));
	checkCudaErrors(cudaMalloc((void **)&d_res, mem_size_res));
	checkCudaErrors(cudaMemset(d_res, 0, mem_size_res));

	// copy host data to device
  	checkCudaErrors(cudaEventRecord(start));
	checkCudaErrors(cudaMemcpy(d_a, h_a, mem_size_a, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_b, h_b, mem_size_b, cudaMemcpyHostToDevice));

	// k_hellowWorld<<<grid, threads>>>();
	// std::cout << "b.x\tb.y\tb.z\tt.x\tk\ta_idx\tb_idx\tres_idx\t\n";
	k_createPts<<<grid, threads>>>(d_a, d_b, d_res, num_a, qtd_b, 0);
	// k_createPts<<<grid, threads>>>(d_a, d_b, d_res, 1, 12, 0);
	// for(int i = 0; i < qtd_a; i += num_a){
	// 	int diff = qtd_a - i;
	// 	if(diff < num_a){
	// 		k_createPts<<<grid, threads>>>(d_a, d_b, d_res, d_nbr, diff, qtd_b, i);
	// 		// cudaMemcpy(h_res, d_res, mem_size_res, cudaMemcpyDeviceToHost);
	// 	} else {
	// 		k_createPts<<<grid, threads>>>(d_a, d_b, d_res, d_nbr, num_a, qtd_b, i);
	// 		// cudaMemcpy(h_res, d_res, mem_size_res, cudaMemcpyDeviceToHost);
	// 	}
	// }

	cudaDeviceSynchronize();
  	checkCudaErrors(cudaEventRecord(stop));
  	checkCudaErrors(cudaEventSynchronize(stop));

  	float msecTotal = 0.0f;
  	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	std::cout << "a.size(): " << a.size() << ", b.size(): " << b.size() << std::endl;
	std::cout << "qtd_a: " << qtd_a << ", qtd_b: " << qtd_b << std::endl;
	std::cout << "num_threads: " << num_threads << ", num_blocks: " << num_blocks << std::endl;
	std::cout << "grid: " << grid.x << ", " << grid.y << ", " << grid.z << std::endl;
	std::cout << "threads: " << threads.x << ", " << threads.y << ", " << threads.z << std::endl;
	std::cout << "mem_size_a: " << mem_size_a << ", mem_size_b: " << mem_size_b << ", mem_size_res: " << mem_size_res << std::endl;
	std::cout << "mem_size_a (MB): " << ((float)mem_size_a)/1024.0/1024.0 << ", mem_size_b (MB): " << ((float)mem_size_b)/1024.0/1024.0 << ", mem_size_res (MB): " << ((float)mem_size_res)/1024.0/1024.0 << std::endl;
	std::cout << "Time: " << msecTotal << std::endl;

	// check if kernel execution generated and error
	getLastCudaError("Kernel execution failed");

	// copy results from device to host
	// checkCudaErrors(cudaMemcpy(h_pts, d_pts, mem_size_pts, cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(h_res, d_res, mem_size_res, cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(h_nbr, d_nbr, mem_size_nbr, cudaMemcpyDeviceToHost));

	// cleanup device memory
	checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaFree(d_b));
	// checkCudaErrors(cudaFree(d_pts));
	checkCudaErrors(cudaFree(d_res));
	// checkCudaErrors(cudaFree(d_nbr));


	// std::cout << "A: " << std::endl;
	// for(int i = 0; i < qtd_a * __SIZE_A; i+=__SIZE_A){
	// 	for(int j = 0; j < __SIZE_A; j++){
	// 		std::cout << h_a[i + j] << ", ";
	// 	}
	// 	std::cout << std::endl;
	// }

	// std::cout << std::endl << "B: " << std::endl;
	// for(int i = 0; i < qtd_b * __SIZE_B; i+=__SIZE_B){
	// 	for(int j = 0; j < __SIZE_B; j++){
	// 		std::cout << h_b[i + j] << ", ";
	// 	}
	// 	std::cout << std::endl;
	// }

	// std::cout << std::endl << "Res: " << std::endl;
	// for(int i = 0; i < NConn; i++){
	// 	for(int j = 0; j < qtd_a; j++){
	// 		for(int k = 0; k < qtd_b; k++){
	// 			int baseIdx = ((i * qtd_a + j) * qtd_b + k) * __SIZE_PTS;
	// 			std::cout << "NConn: " << i << ", a: " << j << ", b: " << k << ", idx: " << baseIdx << " :     ";
	// 			for(int l = 0; l < __SIZE_PTS; l++, baseIdx++){
	// 				std::cout << h_pts[baseIdx] << ", ";
	// 			}
	// 			std::cout << std::endl;
	// 		}
	// 	}
	// }

// #ifdef OPENCV_ENABLED 
// 	int i = 0;
// 	const int max_i = (mem_size_pts / sizeof(int16_t))  - __SIZE_PTS;
// 	std::vector<int16_t> PtsX(__SIZE_PTS/2, 0);
// 	std::vector<int16_t> PtsY(__SIZE_PTS/2, 0);
// 	while(i <= max_i){
// 		for(int j = 0; j < __SIZE_PTS; j+=2){
// 			PtsX[j/2] = h_pts[i + j];
// 			PtsY[j/2] = h_pts[i + j + 1];
// 		}
		
// 		std::cout << "i: " << i << ", i_idx: " << i / __SIZE_PTS << ", res_idx: " << (i / __SIZE_PTS) * __SIZE_RES << ", res: " << h_res[(i / __SIZE_PTS) * __SIZE_RES] << std::endl;
// 		int c = CVHelper::showLayoutMove(PtsX, PtsY);
// 		// i +=__SIZE_PTS * qtd_b * c;
// 		i += __SIZE_PTS*c;
// 		if(i < 0)
// 			i = 0;
// 	}
// #endif


	// cleanup host memory
	// free(h_pts);
	// free(h_res);
	checkCudaErrors(cudaFreeHost(h_res));
}