#include <iostream>
#include <fstream>
#include <vector>
#include "../lib/storage.h"
#include "../lib/globals.h"

Storage::Storage(){
    readConfigs();
}

void Storage::printRoom(RoomConfig room){
    std::cout << room.id << " " << room.name << ": H (" << room.minH << " - " << room.maxH << "), : W (" << room.minW << " - " << room.maxW << "), E: " << room.numExtensions << std::endl;
}

void Storage::readConfigs(){
    setups.clear();
    
    std::ifstream input_file("../configs/rooms", std::ios::binary);
    
    int numOfRooms = 0;

    input_file.read((char*)&numOfRooms, sizeof(int));  
    std::cout << "getConfigs numOfRooms: " << numOfRooms << std::endl;
    
    RoomConfig* rooms = (RoomConfig*)calloc(numOfRooms, sizeof(RoomConfig));
    
    for(int i = 0; i < numOfRooms; i++){
        input_file.read((char*)&(rooms[i].id), sizeof(long));
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
    for (std::vector<RoomConfig>::iterator it = setups.begin() ; it != setups.end(); ++it)
        printRoom(*it);

}

std::vector<RoomConfig> Storage::getConfigs(){
    return setups;
}