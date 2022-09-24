#include <iostream>
#include <fstream>
#include <vector>
#include "../lib/storage.h"
#include "../lib/globals.h"
#include "../lib/log.h"

/// @brief          Storage Constructor
/// @return         None
Storage::Storage(){
    readConfigs();
}

/// @brief          Loads the rooms file and set the private vector "setups" with the rooms information
/// @return         None
void Storage::readConfigs(){
    setups.clear();
    
    std::ifstream input_file("../FloorPlanGenerator/configs/rooms", std::ios::binary);
    
    int numOfRooms = 0;

    input_file.read((char*)&numOfRooms, sizeof(int));  
    std::cout << "getConfigs numOfRooms: " << numOfRooms << std::endl;
    
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

    Log log = Log();
    for (std::vector<RoomConfig>::iterator it = setups.begin() ; it != setups.end(); ++it)
        log.print((RoomConfig)(*it));

}

/// @brief          Get the possible RoomConfig informations
/// @return         RoomConfig vector
std::vector<RoomConfig> Storage::getConfigs(){
    return setups;
}