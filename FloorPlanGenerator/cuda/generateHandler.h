#ifndef GENERATE_HANDLER
#define GENERATE_HANDLER

#include "../lib/globals.h"
#include "../lib/storage.h"
#include <vector>

class GenerateHandler{
private:
    static int checkDefineValues(const std::vector<RoomConfig>& rooms, const int reqSize);

    static int checkThreadCountValue(const int qtdThreadsY);

    static int minThreadCount();

    void consume(std::vector<int16_t> result, Storage& hdd, const int combid, const int taskCount);

public:
    GenerateHandler();

    void generate(
        const std::vector<RoomConfig>& rooms, 
        std::vector<int> allReqCount,
        std::vector<int>& allReq, 
        const int reqSize,
        const int combid,
        Storage& hdd);
};

#endif //GENERATE_HANDLER