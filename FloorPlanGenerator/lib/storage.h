#ifndef STORAGE
#define STORAGE

#include "globals.h"
#include <vector>

/**
  Handles all file managing (read/write) for the project
*/
class Storage
{
    /**
      Vector for all informations baout the rooms setups
    */ 
    std::vector<RoomConfig> setups;

    /// @brief          Loads the rooms file and set the private vector "setups" with the rooms information
    /// @return         None
    void readConfigs();

public:

    /// @brief          Storage Constructor
    /// @return         None
    Storage();

    /// @brief          Get the possible RoomConfig informations
    /// @return         RoomConfig vector
    std::vector<RoomConfig> getConfigs();
};

#endif //STORAGE