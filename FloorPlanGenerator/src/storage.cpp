#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <string>
#include "../lib/storage.h"
#include "../lib/globals.h"
#include "../lib/log.h"
#include "../lib/iter.h"
#include "../lib/calculator.h"

// ls -la ../FloorPlanGenerator/storage/ --block-size=MB
// du -hs ../FloorPlanGenerator/storage/

/// @brief          Storage Constructor
/// @return         None
Storage::Storage(){
    updateProjectDir();
    readConfigs();
}

    
/** 
 * @brief get the project directory
 * @details returns the current executable directory until the first appearence of the folder "FloorPlanGenerator"
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
        input_file.read((char*)&(rooms[i].rPlannyId), sizeof(tempRoom.rPlannyId));
        input_file.read((char*)&(rooms[i].name), ROOM_NAME_SIZE * sizeof(char));
        setups.push_back(rooms[i]);
    }

    input_file.close();
    free(rooms);
}

/// @brief          Get the possible RoomConfig informations
/// @return         RoomConfig vector
std::vector<RoomConfig> Storage::getConfigs(){
    return setups;
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
void Storage::saveResult(const std::vector<std::vector<std::vector<int16_t>>>& layouts, const std::vector<RoomConfig>& rooms, const int n){
    
    int combId  = 0;
    for(int i = 0; i < n; i++){
        combId += rooms[i].id;
    }

    // std::vector<int> qtdSizesH; qtdSizesH.reserve(n);
    // std::vector<int> qtdSizesW; qtdSizesW.reserve(n);
    // for(int i = 0; i < n; i++){
    //     const int diffH = rooms[i].maxH - rooms[i].minH;
    //     const int diffW = rooms[i].maxW - rooms[i].minW;
    //     qtdSizesH.push_back((diffH + rooms[i].step  + rooms[i].step - 1) / rooms[i].step);
    //     qtdSizesW.push_back((diffW + rooms[i].step  + rooms[i].step - 1) / rooms[i].step);
    // }

    std::string path = _projectDir + "/FloorPlanGenerator/storage/" + std::to_string(combId) + ".dat";
    std::ofstream outputFile(path, std::ios::out | std::ios::binary);

    std::cout << "path: " << path << std::endl;

    // int sizeH = 0, sizeW = 0;
    size_t sizeElem = sizeof(layouts[0][0][0]);
    const int sizeLayout = n * sizeElem;
    const int NSizes = layouts.size();
    const int NPerm = layouts[0].size();
    const int NConns = Calculator::NConnections(n) / 2;
    
    const int qtdSizesPerSave = (NConns * NPerm * sizeLayout > 16384) ? 1 : 16384 / (NConns * NPerm * sizeLayout);


    std::cout << "NSizes: " << NSizes << ", NPerm: " << NPerm << ", NConns: " << NConns << ", qtdSizesPerSave: " << qtdSizesPerSave << ", sizeLayout: " << sizeLayout << std::endl;
    // std::vector<int> res; res.reserve();
    // int *sizesFile = (int*)calloc(NPerm*sizeLayout, sizeof(int));
    const int vectorSize = qtdSizesPerSave * NConns * NPerm * sizeLayout;
    std::vector<int16_t> sizesFile; sizesFile.reserve(vectorSize);

    // int count = 0;
    for(int i = 0; i < NSizes; i++){
        // getSizeId(i, n, qtdSizesH, qtdSizesW, &sizeH, &sizeW);
        // const int sizeH_1 = int(sizeH >> 32);
        // const int sizeH_2 = int(sizeH & 0b11111111111111111111111111111111);
        // const int sizeW_1 = int(sizeW >> 32);
        // const int sizeW_2 = int(sizeW & 0b11111111111111111111111111111111);
        
        for(int j = 0; j < NPerm; j++){
            for(long unsigned int k = 0; k < layouts[i][j].size(); k++){
                // sizesFile.push_back(sizeH_1);
                // sizesFile.push_back(sizeH_2);
                // sizesFile.push_back(sizeW_1);
                // sizesFile.push_back(sizeW_2);
                // sizesFile.push_back(sizeH);
                // sizesFile.push_back(sizeW);
                // sizesFile.push_back(j);
                sizesFile.push_back(layouts[i][j][k]);
                // std::cout << layouts[i][j][k] << ", "; 
            }
            // std::cout << std::endl;
        }

        if(i % qtdSizesPerSave == qtdSizesPerSave - 1){
            outputFile.write((char*)&sizesFile[0], sizesFile.size() * sizeof(sizesFile[0]));

            sizesFile.clear(); 
            sizesFile.reserve(vectorSize);
        }
    }
    outputFile.close();
}

std::vector<int> Storage::getSavedCombinations() {
    std::vector<int> result;
    std::string path = _projectDir + "/FloorPlanGenerator/storage";

    for (const auto & entry : std::filesystem::directory_iterator(path)){
        std::string fileName = entry.path().stem();
        std::string extension = entry.path().extension();

        if(extension.compare(".dat") == 0){
            result.push_back(stoi(fileName));
        }
        // std::cout << fileName << "  " << extension <<std::endl;
    }

    return result;
}

// https://stackoverflow.com/questions/15138353/how-to-read-a-binary-file-into-a-vector-of-unsigned-chars
std::vector<int16_t> Storage::readCoreData(int id){
    const std::string filename = _projectDir + "/FloorPlanGenerator/storage/" + std::to_string(id) + ".dat";
    
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);

    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg() / sizeof(int16_t);
    file.seekg(0, std::ios::beg);



    // read the data:
    std::vector<int16_t> fileData(fileSize);
    file.read((char*) &fileData[0], fileSize * sizeof(int16_t));

    return fileData;
}