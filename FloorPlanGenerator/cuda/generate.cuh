#ifndef CUDA_GENERATE
#define CUDA_GENERATE

#include "../lib/globals.h"
#include <vector>

class CudaGenerate
{
private:
    

public:
    static int* createDeviceRoomConfigsArray(const std::vector<RoomConfig>& rooms);

    static int* createDevicePermArray();

    static int* createDeviceAdjArray(
        const std::vector<RoomConfig>& rooms, 
        std::vector<int> allReq, 
        std::vector<int> allReqCount);

    static int* createDeviceAdjCountArray(const std::vector<RoomConfig>& rooms);

    static int16_t* createDeviceResArray(const size_t result_mem_size);

    // static int16_t** createHostResArray(const size_t result_mem_size, const int nThreads);

    static void launchGenereteKernel(
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
        const size_t result_mem_size);

    static void launchDuplicateCheckKernel(int16_t* d_res, int16_t* h_res,const long layouts_count, const size_t result_mem_size);

    static void freeDeviceArrays(int* d_configs, int* d_perm, int* d_adj, int* d_adj_count, int16_t* d_res);

    static void freeHostArrays(int16_t** h_res, const int nThreads);
};

#endif //CUDA_GENERATE