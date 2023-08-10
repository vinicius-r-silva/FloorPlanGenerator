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
		std::vector<RoomConfig> setupsA, std::vector<RoomConfig> setupsB,
    	std::vector<int> allReq);
};

#endif //GPU_HANDLER