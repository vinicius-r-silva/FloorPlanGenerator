#ifndef CUDA_GENERATE
#define CUDA_GENERATE

#include "../lib/globals.h"
#include <vector>

class CudaGenerate
{
private:
    static int* createDeviceRoomConfigsArray(const std::vector<RoomConfig>& rooms);

    static int* createDevicePermArray();
    
    static int8_t* generateSizes(int* d_configs, const long qtdSizes);

public:
    static void generateCuda(const std::vector<RoomConfig>& rooms);
};

#endif //CUDA_GENERATE