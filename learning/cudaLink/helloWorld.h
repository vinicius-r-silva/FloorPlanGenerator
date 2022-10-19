#ifndef HELLOWORLD_CUDA
#define HELLOWORLD_CUDA

class Cuda_test
{
public:

    Cuda_test();
	int launchGPU();
	__host__ __device__ void printHelloGPU();
};

#endif //HELLOWORLD_CUDA