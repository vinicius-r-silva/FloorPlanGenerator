#ifndef GPU_HANDLER
#define GPU_HANDLER

#include "../lib/globals.h"
#include <vector>

class gpuHandler
{

public:
    gpuHandler();
    static void createPts(
        const std::vector<int16_t>& a, const std::vector<int16_t>& b,
    	std::vector<int> allReq, std::string resultPath, int id_a, int id_b);
};

#endif //GPU_HANDLER