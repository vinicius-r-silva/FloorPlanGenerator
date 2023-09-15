#ifndef GPU_HANDLER
#define GPU_HANDLER

#include "../lib/globals.h"
#include <vector>

class CudaCombine
{

public:
    CudaCombine();

    static int* createDeviceAdjArray(const std::vector<int>& allReqAdj);

    static int16_t* createDeviceCoreLayoutsArray(const std::vector<int16_t>& pts);

    static int* createDeviceResArray(const size_t result_mem_size);

    static void freeDeviceArrays(int* adj, int* res, int16_t* a, int16_t* b);

    static void createPts(
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
		int16_t* d_a,
		int16_t* d_b);
};

#endif //GPU_HANDLER