#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <string>
#include <cmath>
#include <set>
#include "../lib/storage.h"
#include "../lib/globals.h"
#include "../lib/log.h"
#include "../lib/iter.h"
#include "../lib/calculator.h"

// ls -la ../FloorPlanGenerator/storage/core --block-size=MB
// du -hS ../FloorPlanGenerator/storage/core

// ls -la ../FloorPlanGenerator/storage/combined/parts --block-size=MB
// du -hS ../FloorPlanGenerator/storage/combined/parts

// ls -la ../FloorPlanGenerator/storage/combined --block-size=MB
// du -hS ../FloorPlanGenerator/storage/combined

// ls -la ../FloorPlanGenerator/storage_prod/core --block-size=MB
// du -hS ../FloorPlanGenerator/storage_prod/core

// ls -la ../FloorPlanGenerator/storage_prod/combined/parts --block-size=MB
// du -hS ../FloorPlanGenerator/storage_prod/combined/parts

// ls -la ../FloorPlanGenerator/storage_prod/combined --block-size=MB
// du -hS ../FloorPlanGenerator/storage_prod/combined

// ls -la ../FloorPlanGenerator/storage_prod/storage_2/core --block-size=MB
// du -hS ../FloorPlanGenerator/storage_prod/storage_2/core

// ls -la ../FloorPlanGenerator/storage_prod/storage_2/combined/parts --block-size=MB
// du -hS ../FloorPlanGenerator/storage_prod/storage_2/combined/parts

// ls -la ../FloorPlanGenerator/storage_prod/storage_2/combined --block-size=MB
// du -hS ../FloorPlanGenerator/storage_prod/storage_2/combined


/// @brief          Storage Constructor
/// @return         None
Storage::Storage(){
    updateProjectDir();
    readConfigs();
    readAdjValues();
    readReqAdjValues();
    // readCombinationResultFiles();
    // readCombinationResultPartFiles();
}

void Storage::updateCombinationList(){
    readCombinationResultFiles();
    readCombinationResultPartFiles();
}
    
/** 
 * @brief get the project directory
 * @details updates the private var "_projectDir" with the current project directory.
 * Iterates over the current executable directory until the first appearence of the folder "FloorPlanGenerator"
 * @return None
*/
void Storage::updateProjectDir(){
    std::string res = ""; //Result string
    const std::filesystem::path currPath = std::filesystem::current_path();
    
    //Iterate over every folder of the current path until it reachs the FloorPlanGenerator folder
    for (auto it = currPath.begin(); it != currPath.end(); ++it){
        res += (*it);
        res += "/";

        if (res.find("FloorPlanGenerator") != std::string::npos) 
            break;
    }

    //Cleanup result
    if (res.rfind("//", 0) == 0)
        res.erase(0,1);

    if(res.length() > 0)
        res.pop_back();

    _projectDir = res;
    // _projectDir += "/Documents/FloorPlanGenerator";
}

std::string Storage::getStoragePath(){
    #ifdef PROD_STORAGE
        // return _projectDir + "/FloorPlanGenerator/storage_prod";
        // return _projectDir + "/FloorPlanGenerator/storage_prod/storage_2";
        return "/home/ribeiro/bigdisk4/step1";
    #else
        return _projectDir + "/FloorPlanGenerator/storage";
    #endif
}

/// @brief  Returns the system path for the cudaResult results folder
/// @return result folder path as string
std::string Storage::getResultPath(){
    return getStoragePath() + "/cudaResult";
}

/// @brief  Returns the system path for the Images folder
/// @return mages folder path as string
std::string Storage::getImagesPath(){
    return getStoragePath() + "/images";
}

/// @brief          Loads the rooms file and set the private vector "setups" with the rooms information
/// @return         None
void Storage::readConfigs(){
    setups.clear();

    std::string path = _projectDir + "/configs/rooms";
    std::ifstream input_file(path, std::ios::binary);
    
    int numOfRooms = 0;

    input_file.read((char*)&numOfRooms, sizeof(int));  
    // std::cout << "getConfigs numOfRooms: " << numOfRooms << std::endl;
    RoomConfig tempRoom;
    
    RoomConfig* rooms = (RoomConfig*)calloc(numOfRooms, sizeof(RoomConfig));
    
    for(int i = 0; i < numOfRooms; i++){
        input_file.read((char*)&(rooms[i].id), sizeof(tempRoom.id));
        input_file.read((char*)&(rooms[i].step), sizeof(tempRoom.step));
        input_file.read((char*)&(rooms[i].numExtensions), sizeof(tempRoom.numExtensions));
        input_file.read((char*)&(rooms[i].minH), sizeof(tempRoom.minH));
        input_file.read((char*)&(rooms[i].maxH), sizeof(tempRoom.maxH));
        input_file.read((char*)&(rooms[i].minW), sizeof(tempRoom.minW));
        input_file.read((char*)&(rooms[i].maxW), sizeof(tempRoom.maxW));
        input_file.read((char*)&(rooms[i].minArea), sizeof(tempRoom.minArea));
        input_file.read((char*)&(rooms[i].maxArea), sizeof(tempRoom.maxArea));
        input_file.read((char*)&(rooms[i].depend), sizeof(tempRoom.depend));
        input_file.read((char*)&(rooms[i].familyIds), sizeof(tempRoom.familyIds));
        input_file.read((char*)&(rooms[i].rPlannyId), sizeof(tempRoom.rPlannyId));
        // input_file.read((char*)&(rooms[i].nameId), sizeof(tempRoom.nameId));
        input_file.read((char*)&(rooms[i].name), ROOM_NAME_SIZE * sizeof(char));
        setups.push_back(rooms[i]);
    }

    input_file.close();
    free(rooms);
}

/// @brief          Loads the adj file and set the private vector "adj_values"
/// @return         None
void Storage::readAdjValues(){
    adj_values.clear();

    std::string path = _projectDir + "/configs/adj";
    std::ifstream input_file(path, std::ios::binary);
    
    int arraySize = 0;

    input_file.read((char*)&arraySize, sizeof(int));      
    for(int i = 0; i < arraySize; i++){
        int value = 0;
        input_file.read((char*)&value, sizeof(value));
        adj_values.push_back(value);
    }

    input_file.close();
}

/// @brief          Loads the adj file and set the private vector "adj_values"
/// @return         None
void Storage::readReqAdjValues(){
    reqadj_values.clear();

    std::string path = _projectDir + "/configs/reqadj";
    std::ifstream input_file(path, std::ios::binary);
    
    int arraySize = 0;
    input_file.read((char*)&arraySize, sizeof(int));      
    const int length = sqrt(arraySize);
    
    if(length * length != arraySize){
        std::cout << "! ERROR. Req Adj MAtrix not square" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // int value = 0;
    // reqadj_values = boost::numeric::ublas::matrix<int>(length, length);
    // for(int i = 0; i < length; i++){
    //     for(int j = 0; j < length; j++){
    //         input_file.read((char*)&value, sizeof(value));
    //         reqadj_values(i,j) = value;
    //     }
    // }

    for(int i = 0; i < arraySize; i++){
        int value = 0;
        input_file.read((char*)&value, sizeof(value));
        reqadj_values.push_back(value);
    }

    input_file.close();
}

std::vector<std::string> Storage::getFoldersInPath(std::string path) {
    std::vector<std::string> result;
    for(const auto & entry : std::filesystem::directory_iterator(path)) {
        if(std::filesystem::is_directory(entry.status())) {
            // std::cout << "getFoldersInPath " << entry.path().filename() << std::endl;
            result.push_back(entry.path().filename());
        }
    }

    return result;
}

void Storage::readCombinationResultPartFiles(){
    std::string path = getStoragePath() + "/combined_parts";
    combinationResultsParts.clear();

    std::vector<std::string> combFolders = getFoldersInPath(path);
    for(std::string combFolder : combFolders) {
        std::stringstream ss(combFolder);
        std::string sCombId, sCombFileId;

        std::getline(ss, sCombId, '_');
        std::getline(ss, sCombFileId);

        int combId = std::stoi(sCombId);
        int combFileId = std::stoi(sCombFileId);
        
        std::string combPath = path + "/" + combFolder;
        std::vector<std::string> minSizeFolders = getFoldersInPath(combPath);
        for(std::string minSizeFolder : minSizeFolders) {
            int minSizeId = std::stoi(minSizeFolder);

            std::string minSizePath = combPath + "/" + minSizeFolder;
            std::vector<std::string> maxSizeFolders = getFoldersInPath(minSizePath);
            for(std::string maxSizeFolder : maxSizeFolders) {
                int maxSizeId = std::stoi(maxSizeFolder);

                std::string maxSizePath = minSizePath + "/" + maxSizeFolder;
                for (const auto & entry : std::filesystem::directory_iterator(maxSizePath)) {
                    std::string fileName = entry.path().stem();
                    std::string extension = entry.path().extension();

                    if(extension.compare(".dat") != 0)
                        continue;

                    int fileId = std::stoi(fileName);

                    CombinationResultPart item(combId, combFileId, minSizeId, maxSizeId, fileId);
                    combinationResultsParts.push_back(item);
                }
            }
        }
    }
}

void Storage::readCombinationResultFiles(){
    std::string path = getStoragePath() + "/combined";
    combinationResults.clear();
    
    std::vector<std::string> combFolders = getFoldersInPath(path);
    for(std::string combFolder : combFolders) {
        std::stringstream ss(combFolder);
        std::string sCombId, sCombFileId;

        std::getline(ss, sCombId, '_');
        std::getline(ss, sCombFileId);

        int combId = std::stoi(sCombId);
        int combFileId = std::stoi(sCombFileId);
        
        std::string combPath = path + "/" + combFolder;
        for (const auto & entry : std::filesystem::directory_iterator(combPath)){
            std::string fileName = entry.path().stem();
            std::string extension = entry.path().extension();

            if(extension.compare(".dat") != 0)
                continue;
                
            std::stringstream sss(fileName);
            std::string sMinSizeId, sMaxSizeId;

            std::getline(sss, sMinSizeId, '_');
            std::getline(sss, sMaxSizeId);

            int minSizeId = std::stoi(sMinSizeId);
            int maxSizeId = std::stoi(sMaxSizeId);

            CombinationResult item(combId, combFileId, minSizeId, maxSizeId);
            combinationResults.push_back(item);
        }
    }
    // // std::cout << "readCombinationResultFiles" << std::endl;
    // for (const auto & entry : std::filesystem::directory_iterator(path)){
    //     std::string fileName = entry.path().stem();
    //     std::string extension = entry.path().extension();

    //     if(extension.compare(".dat") != 0)
    //         continue;
            
    //     std::stringstream ss(fileName);
    //     std::string combId, combFileId, minSizeId, maxSizeId;

    //     std::getline(ss, combId, '_');
    //     std::getline(ss, combFileId, '_');
    //     std::getline(ss, minSizeId, '_');
    //     std::getline(ss, maxSizeId, '.');

    //     // if((std::stoi(minSizeId) >>__RES_FILE_LENGHT_BITS) <= 40)
    //     //     std::cout << "\t\t";
    //     // if(std::stoi(combId) == 3211278)
    //     //     std::cout << fileName << ", combId: " << combId << ", combFileId: " << combFileId << ", min h: " << (std::stoi(minSizeId) >>__RES_FILE_LENGHT_BITS) << ", min w: " << (std::stoi(minSizeId) &__RES_FILE_LENGHT_AND_RULE) << ", max h: " << (std::stoi(maxSizeId) >>__RES_FILE_LENGHT_BITS) << ", max w: " << (std::stoi(maxSizeId) &__RES_FILE_LENGHT_AND_RULE) << std::endl;

    //     CombinationResult item(std::stoi(combId), std::stoi(combFileId), std::stoi(minSizeId), std::stoi(maxSizeId));
    //     combinationResults.push_back(item);
    // }
}

/// @brief          Get the possible RoomConfig informations
/// @return         RoomConfig vector
std::vector<RoomConfig> Storage::getConfigs(){
    return setups;
}

std::vector<RoomConfig> Storage::getConfigsById(int configId){
    std::vector<RoomConfig> result;

    for(RoomConfig room : setups){
        if(room.id & configId){
            result.push_back(room);
        }
    }

    return result;
}

/// @brief          Get the adjacency values
/// @return         int vector
std::vector<int> Storage::getAdjValues(){
    return adj_values;
}

/// @brief          Get the adjacency values
/// @return         int vector
std::vector<int> Storage::getReqAdjValues(){
    return reqadj_values;
}

inline void getSizeId(int k, const int n, const std::vector<int>& qtdSizesH, const std::vector<int>& qtdSizesW, int *idH, int *idW){
    int resH = 0;
    int resW = 0;
    for(int i = 0; i < n; i++){
        resH += (k % qtdSizesH[i]) << (i * 6);
        k /= qtdSizesH[i];

        resW += (k % qtdSizesW[i]) << (i * 6);
        k /= qtdSizesW[i];
    }
    *idH = resH;
    *idW = resW;
}

// layout: size -> order -> conn
void Storage::saveResult(std::vector<int16_t>& layouts, const int combId, const int offsetId){
    if(layouts.size() == 0)
        return;

    std::string path = getStoragePath() + "/core/" + std::to_string(combId) + "_" + std::to_string(offsetId) + ".dat";
    Storage::saveVector(path, layouts);
}

void Storage::saveCombineResultPart(std::vector<std::vector<std::vector<int>>> results, const int combId, const int combFilesdId, const int kernelCount, std::vector<std::vector<int>> fileMaxH, std::vector<std::vector<int>> fileMaxW){
    std::string combFolder = getStoragePath() + "/combined_parts/" + std::to_string(combId) + "_" + std::to_string(combFilesdId);
    if (!std::filesystem::exists(combFolder) || !std::filesystem::is_directory(combFolder)) {
        std::filesystem::create_directory(combFolder);
    }
    
    const int max_size = results.size();
    for(int h = 0; h < max_size; h++){
        for(int w = 0; w < max_size; w++){
            if(results[h][w].size() == 0)
                continue;

            int minSizeId = (h << __RES_FILE_LENGHT_BITS) | w;
            int maxSizeId = (fileMaxH[h][w] << __RES_FILE_LENGHT_BITS) | fileMaxW[h][w];

            std::string minSizeFolder = combFolder + "/" + std::to_string(minSizeId); 
            std::string maxSizeFolder = minSizeFolder + "/" + std::to_string(maxSizeId); 
            if (!std::filesystem::exists(maxSizeFolder) || !std::filesystem::is_directory(maxSizeFolder)) {
                if (!std::filesystem::exists(minSizeFolder) || !std::filesystem::is_directory(minSizeFolder)) {
                    std::filesystem::create_directory(minSizeFolder);
                }
                std::filesystem::create_directory(maxSizeFolder);
            }

            std::string filePath = maxSizeFolder + "/" + std::to_string(kernelCount) + ".dat";

            // long totalSize = results[i][w].size();
            // std::cout << combId << ", " << minSizeId << ", " << kernelCount << ", " << totalSize << ", " << totalSize / __SIZE_RES_DISK << ", " << ((double)(totalSize * sizeof(int))) / 1024.0 / 1024.0 << std::endl;

            // std::cout << "i: " << i << ", w: " << w << ", minSizeId: " << minSizeId << std::endl;
            // std::string path = getStoragePath() + "/combined/parts/" + std::to_string(combId) + "_" + std::to_string(combFilesdId) + "_" + std::to_string(minSizeId) + "_" + std::to_string(maxSizeId) + "_" + std::to_string(kernelCount) + ".dat";
            
            Storage::saveVector(filePath, results[h][w]);
        }
    }
}

void Storage::saveCombineResult(std::vector<int> results, const int combId, const int combFileId, const int minSizeId, const int maxSizeId){
    std::string combFolder = getStoragePath() + "/combined/" + std::to_string(combId) + "_" + std::to_string(combFileId);
    if (!std::filesystem::exists(combFolder) || !std::filesystem::is_directory(combFolder)) {
        std::filesystem::create_directory(combFolder);
    }

    std::string filePath = combFolder + "/" + std::to_string(minSizeId) + "_" + std::to_string(maxSizeId) + ".dat";
    
    // std::string path = getStoragePath() + "/combined/" + std::to_string(combId) + "_" + std::to_string(combFileId) + "_" + std::to_string(minSizeId) + "_" + std::to_string(maxSizeId) + ".dat";
    Storage::saveVector(filePath, results);
}

template <typename T>
void Storage::saveVector(std::string fullPath, std::vector<T>& arr){    
    size_t elemSize = sizeof(T);
    std::ofstream outputFile(fullPath, std::ios::out | std::ios::binary);
    outputFile.write(reinterpret_cast<const char*>(arr.data()), arr.size() * elemSize);
    outputFile.close();
}

std::vector<int> Storage::getSavedCoreFiles(int id) {
    std::set<int> fileIds;
    std::string id_name = std::to_string(id);
    std::string path = getStoragePath() + "/core";

    for (const auto & entry : std::filesystem::directory_iterator(path)){
        std::string fileName = entry.path().stem();
        std::string extension = entry.path().extension();

        if(extension.compare(".dat") != 0)
            continue;

        if (fileName.find(id_name) == std::string::npos)
            continue;
            
        size_t split_pos = fileName.find('_');
        
        std::string fileId = fileName.substr(split_pos + 1, fileName.length() - 1);
        fileIds.insert(stoi(fileId));
    }

    std::vector<int> result(fileIds.begin(), fileIds.end());
    return result;
}


std::vector<int> Storage::getSavedCores() {
    std::set<int> ids;
    std::string path = getStoragePath() + "/core";

    for (const auto & entry : std::filesystem::directory_iterator(path)){
        std::string fileName = entry.path().stem();
        std::string extension = entry.path().extension();

        if(extension.compare(".dat") != 0)
            continue;

        size_t split_pos = fileName.find('_');
        if (split_pos != std::string::npos){
            if(fileName.find("_0") == std::string::npos)
                continue;
            
            fileName = fileName.substr(0, split_pos);
            ids.insert(stoi(fileName));
        }
    }

    std::vector<int> result(ids.begin(), ids.end());
    return result;
}


std::vector<int> Storage::getSavedCombinationsPartsCombIds() {
    std::set<int> ids;
    for(CombinationResultPart item : combinationResultsParts){
        ids.insert(item.combId);
    }

    std::vector<int> result(ids.begin(), ids.end());
    return result;
}


std::vector<CombinationResultPart> Storage::getSavedCombinationsParts(int combId, int combFileId, int minSizeId) {
    std::vector<CombinationResultPart> result;
    for(CombinationResultPart item : combinationResultsParts){
        if(item.combId == combId&& item.combFileId == combFileId && item.minSizeId == minSizeId)
            result.push_back(item);
    }

    return result;
}


std::vector<int> Storage::getSavedCombinationsPartsCombFileIds(int combId) {
    std::set<int> ids;
    for(CombinationResultPart item : combinationResultsParts){
        if(item.combId == combId)
            ids.insert(item.combFileId);
    }

    std::vector<int> result(ids.begin(), ids.end());
    return result;
}


std::vector<int> Storage::getSavedCombinationsPartsMinSizeIds(int combId, int combFileId) {
    std::set<int> ids;
    for(CombinationResultPart item : combinationResultsParts){
        if(item.combId == combId && item.combFileId == combFileId)
            ids.insert(item.minSizeId);
    }

    std::vector<int> result(ids.begin(), ids.end());
    return result;
}


std::vector<int> Storage::getSavedCombinationsCombIds() {
    std::set<int> ids;
    for(CombinationResult item : combinationResults){
        ids.insert(item.combId);
    }

    std::vector<int> result(ids.begin(), ids.end());
    return result;
}


std::vector<CombinationResult> Storage::getSavedCombinations(int combId, int combFileId) {
    std::vector<CombinationResult> result;
    for(CombinationResult item : combinationResults){
        if(item.combId == combId && item.combFileId == combFileId)
            result.push_back(item);
    }

    return result;
}


std::vector<int> Storage::getSavedCombinationsCombFileIds(int combId) {
    std::set<int> ids;
    for(CombinationResult item : combinationResults){
        if(item.combId == combId)
            ids.insert(item.combFileId);
    }

    std::vector<int> result(ids.begin(), ids.end());
    return result;
}


std::vector<int> Storage::getSavedCombinationsMinSizeIds(int combId, int combFileId) {
    std::set<int> ids;
    for(CombinationResult item : combinationResults){
        if(item.combId == combId && item.combFileId == combFileId)
            ids.insert(item.minSizeId);
    }

    std::vector<int> result(ids.begin(), ids.end());
    return result;
}

// https://stackoverflow.com/questions/15138353/how-to-read-a-binary-file-into-a-vector-of-unsigned-chars
std::vector<int16_t> Storage::readCoreData(const int combId, const int fileId){
    const std::string filename = getStoragePath() + "/core/" + std::to_string(combId) + "_" + std::to_string(fileId) + ".dat";
    std::cout << "readCoreData: " << filename << std::endl;
    return readVector<int16_t>(filename);
}

std::vector<int> Storage::readCombineData(const int combId, const int combFileId, const int minSizeId, const int maxSizeId){
    const std::string filename = getStoragePath() + "/combined/" + std::to_string(combId) + "_" + std::to_string(combFileId) + "/" + std::to_string(minSizeId) + "_" + std::to_string(maxSizeId) + ".dat";
    std::cout << "readCombineData: " << filename << std::endl;
    return readVector<int>(filename);
}

std::vector<int> Storage::readCombineSplitData(const int combId, const int combFileId, const int minSizeId, const int maxSizeId, const int fileId){
    const std::string filename = getStoragePath() + "/combined_parts/" + std::to_string(combId) + "_" + std::to_string(combFileId) + "/" + std::to_string(minSizeId) + "/" + std::to_string(maxSizeId) + "/" + std::to_string(fileId) + ".dat";
    return readVector<int>(filename);
}

template <typename T>
std::vector<T> Storage::readVector(std::string fullPath){    
    // open the file:
    std::streampos fileSize;
    std::ifstream file(fullPath, std::ios::binary);

    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg() / sizeof(T);
    file.seekg(0, std::ios::beg);

    // read the data:
    std::vector<T> fileData(fileSize, -1);
    // file.read((char*) &fileData[0], fileSize * sizeof(T));
    file.read(reinterpret_cast<char*>(fileData.data()), fileData.size() * sizeof(T)); // char==byte


    return fileData;
}


void Storage::deleteSavedCoreResults() {
    std::string path = getStoragePath() + "/core";

    for (const auto & entry : std::filesystem::directory_iterator(path)){
        std::string extension = entry.path().extension();

        if(extension.compare(".dat") == 0){
            remove(entry.path());
        }
    }
}


void Storage::deleteSavedCombinedResultsParts() {
    std::string path = getStoragePath() + "/combined_parts";

    std::vector<std::string> combFolders = getFoldersInPath(path);
    std::cout << "deleteSavedCombinedResultsParts combFolders" << std::endl;
    Log::printVector1D(combFolders);

    int count = 0;
    for(std::string combFolder : combFolders) {        
        std::string combPath = path + "/" + combFolder; 

        std::vector<std::string> minSizeFolders = getFoldersInPath(combPath);
        std::cout << "deleteSavedCombinedResultsParts minSizeFolders" << std::endl;
        Log::printVector1D(minSizeFolders);

        for(std::string minSizeFolder : minSizeFolders) {
            std::string minSizePath = combPath + "/" + minSizeFolder;
            std::vector<std::string> maxSizeFolders = getFoldersInPath(minSizePath);

            for(std::string maxSizeFolder : maxSizeFolders) {
                std::string maxSizePath = minSizePath + "/" + maxSizeFolder;
            
                for (const auto & entry : std::filesystem::directory_iterator(maxSizePath)) {
                    std::string extension = entry.path().extension();

                    if(extension.compare(".dat") == 0){
                        remove(entry.path());
                        count++;
                    }
                }
            }
            std::cout << "deleteSavedCombinedResultsParts count"  << count << std::endl;
        }
    }
    
    readCombinationResultPartFiles();
}


void Storage::deleteSavedCombinedResults() {
    std::string path = getStoragePath() + "/combined";

    std::vector<std::string> combFolders = getFoldersInPath(path);
    std::cout << "deleteSavedCombinedResults combFolders" << std::endl;
    Log::printVector1D(combFolders);

    int count = 0;
    for(std::string combFolder : combFolders) {        
        std::string combPath = path + "/" + combFolder;
        
        for (const auto & entry : std::filesystem::directory_iterator(combPath)) {
            std::string extension = entry.path().extension();

            if(extension.compare(".dat") == 0){
                remove(entry.path());
                count++;
            }
        }

        std::cout << "deleteSavedCombinedResultsParts count "  << count << std::endl;
    }
    // std::string path = getStoragePath() + "/combined";

    // for (const auto & entry : std::filesystem::directory_iterator(path)){
    //     std::string extension = entry.path().extension();

    //     if(extension.compare(".dat") == 0){
    //         remove(entry.path());
    //     }
    // }
    
    readCombinationResultFiles();
    readCombinationResultPartFiles();
}


void Storage::deleteSavedImages() {
    std::string path = getImagesPath();

    for (const auto & entry : std::filesystem::directory_iterator(path)){
        std::string extension = entry.path().extension();

        if(extension.compare(".png") == 0 || extension.compare(".jpg") == 0){
            remove(entry.path());
        }
    }
}


std::vector<int> Storage::getSavedResults() {
    std::vector<int> result;
    std::string path = getStoragePath() + "/cudaResult";

    for (const auto & entry : std::filesystem::directory_iterator(path)){
        std::string fileName = entry.path().stem();
        std::string extension = entry.path().extension();

        if(extension.compare(".dat") == 0){
            result.push_back(stoi(fileName));
        }
    }

    return result;
}

std::vector<int> Storage::readResultData(int combId, int fileId){
    const std::string filename = getStoragePath() + "/core/" + std::to_string(combId) + "_" + std::to_string(fileId) + ".dat";
    
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);

    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg() / sizeof(int);
    file.seekg(0, std::ios::beg);

    // read the data:
    std::vector<int> fileData(fileSize);
    file.read((char*) &fileData[0], fileSize * sizeof(int));

    return fileData;
}


template std::vector<int16_t> Storage::readVector(std::string fullPath);
template std::vector<int> Storage::readVector(std::string fullPath);
template std::vector<size_t> Storage::readVector(std::string fullPath);
template void Storage::saveVector(std::string fullPath, std::vector<int16_t>& arr);
template void Storage::saveVector(std::string fullPath, std::vector<int>& arr);
template void Storage::saveVector(std::string fullPath, std::vector<size_t>& arr);
