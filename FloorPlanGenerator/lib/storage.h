#ifndef STORAGE
#define STORAGE

#include "globals.h"
#include <vector>

class Storage
{
    std::vector<RoomConfig> setups;
    void readConfigs();
    void printRoom(RoomConfig room);

public:
    Storage();
    std::vector<RoomConfig> getConfigs();
};

#endif //STORAGE