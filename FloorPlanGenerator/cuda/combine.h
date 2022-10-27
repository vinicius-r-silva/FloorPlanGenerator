#ifndef GPU_HANDLER
#define GPU_HANDLER

#include <vector>

class gpuHandler
{

public:
    gpuHandler();
    static void createPts(const std::vector<int16_t>& a, const std::vector<int16_t>& b);
};

#endif //GPU_HANDLER