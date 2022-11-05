#ifndef STORAGE
#define STORAGE

// ls -la ../FloorPlanGenerator/storage/ --block-size=MB
// du -hs ../FloorPlanGenerator/storage/

#include "globals.h"
#include <stdint.h>
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
      Vector with multiplicaiton value for each adjcency
    */ 
    std::vector<int> adj_values;
    
    /** 
     * @brief get the project directory
     * @details returns the current executable directory until the first appearence of the folder "FloorPlanGenerator"
     * @return None
    */
    void updateProjectDir();

    /// @brief          Loads the rooms file and set the private vector "setups" with the rooms information
    /// @return         None
    void readConfigs();

    /// @brief          Loads the adj file and set the private vector "adj_values"
    /// @return         None
    void readAdjValues();

public:

    /// @brief          Storage Constructor
    /// @return         None
    Storage();

    /// @brief          Get the possible RoomConfig informations
    /// @return         RoomConfig vector
    std::vector<RoomConfig> getConfigs();

    /// @brief          Get the adjacency values
    /// @return         int vector
    std::vector<int> getAdjValues();

    void saveResult(const std::vector<std::vector<std::vector<int16_t>>>& res, const std::vector<RoomConfig>& rooms, const int n);

    std::vector<int> getSavedCombinations();
    
    std::vector<int16_t> readCoreData(int id);
};

#endif //STORAGE