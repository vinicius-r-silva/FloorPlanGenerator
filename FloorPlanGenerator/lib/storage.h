#ifndef STORAGE
#define STORAGE

// ls -la ../FloorPlanGenerator/storage/ --block-size=MB
// du -hs ../FloorPlanGenerator/storage/

// ls -la ../FloorPlanGenerator/storage/combination --block-size=MB
// du -hs ../FloorPlanGenerator/storage/combination

#include "globals.h"
#include <stdint.h>
#include <vector>
#include <string>
// #include <boost/numeric/ublas/matrix.hpp>

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
      Vector with multiplicaiton value for each adjcency
    */ 
    std::vector<int> reqadj_values;
    // boost::numeric::ublas::matrix<int> reqadj_values;
    
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

    /// @brief          Loads the adj file and set the private vector "adj_values"
    /// @return         None
    void readReqAdjValues();

public:

    /// @brief          Storage Constructor
    /// @return         None
    Storage();

    /// @brief          Returns the system path for the combination results folder
    /// @return         result folder path as string
    std::string getResultPath();

    /// @brief  Returns the system path for the Images folder
    /// @return mages folder path as string
    std::string getImagesPath();

    /// @brief          Get the possible RoomConfig informations
    /// @return         RoomConfig vector
    std::vector<RoomConfig> getConfigs();

    /// @brief          Get the adjacency values
    /// @return         int vector
    std::vector<int> getAdjValues();

    /// @brief          Get the adjacency values
    /// @return         int vector
    std::vector<int> getReqAdjValues();
    // boost::numeric::ublas::matrix<int> getReqAdjValues();

    void saveResult(const std::vector<int16_t>& res, const std::vector<RoomConfig>& rooms, const int n);

    std::vector<int> getSavedCoreCombinations();
    
    std::vector<int16_t> readCoreData(int id);

    template <typename T>
    std::vector<T> readVector(std::string fullPath);

    std::vector<int> getSavedResults();

    std::vector<int> readResultData(int id);
};

#endif //STORAGE