#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <string>
#include <cmath>
#include "../lib/storage.h"
#include "../lib/globals.h"
#include "../lib/log.h"
#include "../lib/iter.h"
#include "../lib/calculator.h"

// ls -la ../FloorPlanGenerator/storage/core --block-size=MB
// du -hs ../FloorPlanGenerator/storage/core

// ls -la ../FloorPlanGenerator/storage/cudaResult --block-size=MB
// du -hs ../FloorPlanGenerator/storage/cudaResult

/// @brief          Storage Constructor
/// @return         None
Storage::Storage(){
    updateProjectDir();
    readConfigs();
    readAdjValues();
    readReqAdjValues();
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

/// @brief  Returns the system path for the cudaResult results folder
/// @return result folder path as string
std::string Storage::getResultPath(){
    return _projectDir + "/FloorPlanGenerator/storage/cudaResult";
}

/// @brief  Returns the system path for the Images folder
/// @return mages folder path as string
std::string Storage::getImagesPath(){
    return _projectDir + "/FloorPlanGenerator/storage/images";
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

/// @brief          Get the possible RoomConfig informations
/// @return         RoomConfig vector
std::vector<RoomConfig> Storage::getConfigs(){
    return setups;
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
void Storage::saveResult(const std::vector<int16_t>& layouts, const int combId, const int offsetId){
    if(layouts.size() == 0)
        return;

    std::string path = _projectDir + "/FloorPlanGenerator/storage/core/" + std::to_string(combId) + "_" + std::to_string(offsetId) + ".dat";
    Storage::saveVector(path, layouts, sizeof(int16_t));
}

template <typename T>
void Storage::saveVector(std::string fullPath, std::vector<T> arr, size_t elemSize){    
    std::ofstream outputFile(fullPath, std::ios::out | std::ios::binary);
    outputFile.write(reinterpret_cast<const char*>(arr.data()), arr.size() * elemSize);
    outputFile.close();
}

std::vector<int> Storage::getSavedCoreCombinationFiles(int id) {
    std::string id_name = std::to_string(id);
    std::vector<int> result;
    std::string path = _projectDir + "/FloorPlanGenerator/storage/core";

    for (const auto & entry : std::filesystem::directory_iterator(path)){
        std::string fileName = entry.path().stem();
        std::string extension = entry.path().extension();


        if(extension.compare(".dat") != 0)
            continue;

        if (fileName.find(id_name) != 0)
            continue;
            
        size_t split_pos = fileName.find('_');
        
        std::string fileId = fileName.substr(split_pos + 1, fileName.length() - 1);
        result.push_back(stoi(fileId));
    }

    return result;
}


std::vector<int> Storage::getSavedCoreCombinations() {
    std::vector<int> result;
    std::string path = _projectDir + "/FloorPlanGenerator/storage/core";

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
            result.push_back(stoi(fileName));
        }
    }

    return result;
}

// https://stackoverflow.com/questions/15138353/how-to-read-a-binary-file-into-a-vector-of-unsigned-chars
std::vector<int16_t> Storage::readCoreData(int combId, int fileId){
    const std::string filename = _projectDir + "/FloorPlanGenerator/storage/core/" + std::to_string(combId) + "_" + std::to_string(fileId) + ".dat";
    return readVector<int16_t>(filename);
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
    std::vector<T> fileData(fileSize);
    file.read((char*) &fileData[0], fileSize * sizeof(T));

    return fileData;
}


void Storage::deleteSavedResults() {
    std::string path = _projectDir + "/FloorPlanGenerator/storage/core";

    for (const auto & entry : std::filesystem::directory_iterator(path)){
        std::string extension = entry.path().extension();

        if(extension.compare(".dat") == 0){
            remove(entry.path());
        }
    }
}


std::vector<int> Storage::getSavedResults() {
    std::vector<int> result;
    std::string path = _projectDir + "/FloorPlanGenerator/storage/cudaResult";

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
    const std::string filename = _projectDir + "/FloorPlanGenerator/storage/core/" + std::to_string(combId) + "_" + std::to_string(fileId) + ".dat";
    
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
template void Storage::saveVector(std::string fullPath, std::vector<int16_t> arr, size_t elemSize);