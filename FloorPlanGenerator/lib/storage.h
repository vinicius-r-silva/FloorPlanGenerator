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
    std::string _projectDir;

    /**
      Vector for all informations baout the rooms setups
    */ 
    std::vector<RoomConfig> setups;
    
    /** 
     * @brief get the project directory
     * @details returns the current executable directory until the first appearence of the folder "FloorPlanGenerator"
     * @return None
    */
    void updateProjectDir();

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

    void saveResult(const std::vector<std::vector<std::vector<int>>>& res, const std::vector<RoomConfig>& rooms, const int n);

    std::vector<int> getSavedCombinations();
    
    std::vector<int> readCoreData(int id);
};

#endif //STORAGE