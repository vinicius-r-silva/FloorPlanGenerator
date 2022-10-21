#ifndef CUDA_COMBINE
#define CUDA_COMBINE

#include <vector>

class Cuda_Combine
{

public:
    Cuda_Combine();
    static int launchGPU(const std::vector<int16_t>& a, const std::vector<int16_t>& b);
};

#endif //HELLOWORLD_CUDA