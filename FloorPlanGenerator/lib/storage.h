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


    // TODO change to map
    std::vector<CombinationResult> combinationResults;

    std::vector<CombinationResultPart> combinationResultsParts;


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

    void readCombinationResultFiles();

    void readCombinationResultPartFiles();

public:

    /// @brief          Storage Constructor
    /// @return         None
    Storage();

    void updateCombinationList();

    std::string getStoragePath();

    /// @brief          Returns the system path for the combination results folder
    /// @return         result folder path as string
    std::string getResultPath();

    /// @brief  Returns the system path for the Images folder
    /// @return mages folder path as string
    std::string getImagesPath();

    /// @brief          Get the possible RoomConfig informations
    /// @return         RoomConfig vector
    std::vector<RoomConfig> getConfigs();

    std::vector<RoomConfig> getConfigsById(int configId);

    /// @brief          Get the adjacency values
    /// @return         int vector
    std::vector<int> getAdjValues();

    /// @brief          Get the adjacency values
    /// @return         int vector
    std::vector<int> getReqAdjValues();
    // boost::numeric::ublas::matrix<int> getReqAdjValues();
 

    void saveResult(std::vector<int16_t>& layouts, const int combId, const int offsetId);

    // void saveCombineResult(const std::vector<int>& result, const std::vector<size_t>& index_table, const int combId, const int offsetId);
    void saveCombineResultPart(std::vector<std::vector<std::vector<int>>> results, const int combId, const int combFilesdId, const int kernelLaunchCount, std::vector<std::vector<int>> fileMaxH, std::vector<std::vector<int>> fileMaxW);

    void saveCombineResult(std::vector<int> results, const int combId, const int combFileId, const int minSizeId, const int maxSizeId);

    std::vector<int> getSavedCores();

    std::vector<int> getSavedCoreFiles(int id);

    std::vector<int> getSavedCombinationsPartsCombIds();

    std::vector<CombinationResultPart> getSavedCombinationsParts(int combId, int combFileId, int minSizeId);

    std::vector<int> getSavedCombinationsPartsCombFileIds(int combId);

    std::vector<int> getSavedCombinationsPartsMinSizeIds(int combId, int combFileId);
    
    std::vector<int> getSavedCombinationsPartsKernelIds(int combId, int combFileId, int minSizeId);

    std::vector<int> getSavedCombinationsCombIds();

    std::vector<CombinationResult> getSavedCombinations(int combId, int combFileId);

    std::vector<int> getSavedCombinationsCombFileIds(int combId);

    std::vector<int> getSavedCombinationsMinSizeIds(int combId, int combFileId);

    std::vector<int16_t> readCoreData(const int combId, const int fileId);

    std::vector<int> readCombineData(const int combId, const int combFileId, const int minSizeId, const int maxSizeId);

    std::vector<int> readCombineSplitData(const int combId, const int combFileId, const int minSizeId, const int maxSizeId, const int fileId);

    template <typename T>
    void saveVector(std::string fullPath, std::vector<T>& arr);

    template <typename T>
    std::vector<T> readVector(std::string fullPath);

    void deleteSavedCoreResults();

    void deleteSavedCombinedResults();

    void deleteSavedCombinedResultsParts();

    void deleteSavedImages();

    std::vector<int> getSavedResults();

    std::vector<int> readResultData(int combId, int fileId);
};

#endif //STORAGE