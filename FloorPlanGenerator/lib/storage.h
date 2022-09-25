#ifndef STORAGE
#define STORAGE

#include "globals.h"
#include <vector>
#include <string>

/**
  Handles all file managing (read/write) for the project
*/
class Storage
{
    /**
      Vector for all informations baout the rooms setups
    */ 
    std::vector<RoomConfig> setups;
    
    /** 
     * @brief get the project directory
     * @details returns the current executable directory until the first appearence of the folder "FloorPlanGenerator"
     * @return String of the current project directory
    */
    std::string getProjectDir();

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