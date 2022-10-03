#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <string>
#include "../lib/storage.h"
#include "../lib/globals.h"
#include "../lib/log.h"

/// @brief          Storage Constructor
/// @return         None
Storage::Storage(){
    readConfigs();
}

    
/** 
 * @brief get the project directory
 * @details returns the current executable directory until the first appearence of the folder "FloorPlanGenerator"
 * @return String of the current project directory
*/
std::string Storage::getProjectDir(){
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

    return res;
}

/// @brief          Loads the rooms file and set the private vector "setups" with the rooms information
/// @return         None
void Storage::readConfigs(){
    setups.clear();

    std::string path = getProjectDir() + "/configs/rooms";
    std::ifstream input_file(path, std::ios::binary);
    
    int numOfRooms = 0;

    input_file.read((char*)&numOfRooms, sizeof(int));  
    // std::cout << "getConfigs numOfRooms: " << numOfRooms << std::endl;
    
    RoomConfig* rooms = (RoomConfig*)calloc(numOfRooms, sizeof(RoomConfig));
    
    for(int i = 0; i < numOfRooms; i++){
        input_file.read((char*)&(rooms[i].id), sizeof(long));
        input_file.read((char*)&(rooms[i].step), sizeof(int));
        input_file.read((char*)&(rooms[i].numExtensions), sizeof(int));
        input_file.read((char*)&(rooms[i].minH), sizeof(int));
        input_file.read((char*)&(rooms[i].maxH), sizeof(int));
        input_file.read((char*)&(rooms[i].minW), sizeof(int));
        input_file.read((char*)&(rooms[i].maxW), sizeof(int));
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