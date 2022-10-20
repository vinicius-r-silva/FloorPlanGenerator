#ifndef CUDA_COMBINE
#define CUDA_COMBINE

#include <vector>

class Cuda_Combine
{

public:
    Cuda_Combine();
	static int launchGPU(const std::vector<int>& a, const std::vector<int>& b, const int n_a, const int n_b);
};

#endif //HELLOWORLD_CUDA