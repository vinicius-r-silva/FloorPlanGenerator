#ifndef COMBINE_HANDLER
#define COMBINE_HANDLER

#include "../lib/globals.h"
#include "../lib/storage.h"
#include <vector>

class CombineHandler{
private:
    static int checkDefineValues(const std::vector<RoomConfig>& a, const std::vector<RoomConfig>& b, std::vector<int> adj);

    static int checkThreadCountValue(const int qtdThreadsY);

    static int minThreadCount();

    // static void drawResult(const int *h_res, const size_t res_mem_size);

    void consume(const std::vector<int>& h_res, const size_t res_mem_size, Storage& hdd, const int combId, const int taskCount, const int max_layout_size);

    std::vector<int> createConns(const std::vector<RoomConfig>& rooms_a, const std::vector<RoomConfig>& rooms_b);

    int getMaxLayoutSize(const std::vector<RoomConfig>& rooms_a, const std::vector<RoomConfig>& rooms_b);

    int getMinLayoutSize(const std::vector<RoomConfig>& rooms_a, const std::vector<RoomConfig>& rooms_b);
    
    int getRoomsCombId(const std::vector<RoomConfig>& rooms);

    void moveElements(std::vector<int>& arr, size_t begin, size_t offset);

public:
    CombineHandler();

    void combine(
        const std::vector<RoomConfig>& rooms_a, 
        const std::vector<RoomConfig>& rooms_b, 
        const std::vector<int16_t>& a, 
        const std::vector<int16_t>& b,
    	std::vector<int> allReqAdj, 
        Storage& hdd);
};

#endif //COMBINE_HANDLER