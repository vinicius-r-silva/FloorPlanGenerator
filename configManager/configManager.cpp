#include <iostream>
#include <fstream>
#include <string.h>
#include <inttypes.h>
#include "../FloorPlanGenerator/lib/globals.h"

void writeRoom(RoomConfig room, std::ofstream& file){
    file.write((char*)&(room.id), sizeof(long));
    file.write((char*)&(room.step), sizeof(int));
    file.write((char*)&(room.numExtensions), sizeof(int));
    file.write((char*)&(room.minH), sizeof(int));
    file.write((char*)&(room.maxH), sizeof(int));
    file.write((char*)&(room.minW), sizeof(int));
    file.write((char*)&(room.maxW), sizeof(int));
    file.write((char*)&(room.name), ROOM_NAME_SIZE * sizeof(char));
}

void printRoom(RoomConfig room){
    std::cout << room.id << " " << room.name << ": H (" << room.minH << " - " << room.maxH << "), : W (" << room.minW << " - " << room.maxW << "), Ext: " << room.numExtensions << ", Step: " << room.step << std::endl;
}

void normalConfig(){
    int numOfRooms = 6;

    RoomConfig *rooms = (RoomConfig*)calloc(numOfRooms , sizeof(RoomConfig));
    memset(rooms[0].name, '\0', ROOM_NAME_SIZE);
    memset(rooms[1].name, '\0', ROOM_NAME_SIZE);
    memset(rooms[2].name, '\0', ROOM_NAME_SIZE);
    memset(rooms[3].name, '\0', ROOM_NAME_SIZE);
    memset(rooms[4].name, '\0', ROOM_NAME_SIZE);
    memset(rooms[5].name, '\0', ROOM_NAME_SIZE);
    

    rooms[0].id = 1 << 0;
    rooms[0].numExtensions = 2;
    rooms[0].name[0] = 's'; rooms[0].name[1] = 'a';
    rooms[0].name[2] = 'l'; rooms[0].name[3] = 'a';
    rooms[0].minH = 30; rooms[0].maxH = 50;
    rooms[0].minW = 20; rooms[0].maxW = 40;
    rooms[0].step = 20;
    
    rooms[1].id = 1 << 1;
    rooms[1].numExtensions = 0;
    rooms[1].name[0] = 'b'; rooms[1].name[1] = 'a';
    rooms[1].name[2] = 'n'; rooms[1].name[3] = 'h';
    rooms[1].name[4] = 'e'; rooms[1].name[5] = 'i';
    rooms[1].name[6] = 'r'; rooms[1].name[7] = 'o';
    rooms[1].minH = 8; rooms[1].maxH = 20;
    rooms[1].minW = 15; rooms[1].maxW = 30;
    rooms[1].step = 20;

    rooms[2].id = 1 << 2;
    rooms[2].numExtensions = 1;
    rooms[2].name[0] = 'q'; rooms[2].name[1] = 'u';
    rooms[2].name[2] = 'a'; rooms[2].name[3] = 'r';
    rooms[2].name[4] = 't'; rooms[2].name[5] = 'o';
    rooms[2].minH = 20; rooms[2].maxH = 40;
    rooms[2].minW = 20; rooms[2].maxW = 40;
    rooms[2].step = 20;
    
    rooms[3].id = 1 << 3;
    rooms[3].numExtensions = 0;
    rooms[3].name[0] = 'c'; rooms[3].name[1] = 'o';
    rooms[3].name[2] = 'r'; rooms[3].name[3] = 'r';
    rooms[3].name[4] = 'e'; rooms[3].name[5] = 'd';
    rooms[3].name[6] = 'o'; rooms[3].name[7] = 'r';
    rooms[3].minH = 7; rooms[3].maxH = 15;
    rooms[3].minW = 7; rooms[3].maxW = 50;
    rooms[3].step = 20;
    
    rooms[4].id = 1 << 4;
    rooms[4].numExtensions = 0;
    rooms[4].name[0] = 'c'; rooms[4].name[1] = 'o';
    rooms[4].name[2] = 'z'; rooms[4].name[3] = 'i';
    rooms[4].name[4] = 'n'; rooms[4].name[5] = 'h';
    rooms[4].name[6] = 'a';
    rooms[4].minH = 15; rooms[4].maxH = 25;
    rooms[4].minW = 15; rooms[4].maxW = 30;
    rooms[4].step = 20;
    
    rooms[5].id = 1 << 5;
    rooms[5].numExtensions = 0;
    rooms[5].name[0] = 'l'; rooms[5].name[1] = 'a';
    rooms[5].name[2] = 'v'; rooms[5].name[3] = 'a';
    rooms[5].name[4] = 'n'; rooms[5].name[5] = 'd';
    rooms[5].name[6] = 'e'; rooms[5].name[7] = 'r';
    rooms[5].name[8] = 'i'; rooms[5].name[9] = 'a';
    rooms[5].minH = 15; rooms[5].maxH = 25;
    rooms[5].minW = 15; rooms[5].maxW = 30;
    rooms[5].step = 20;

    std::ofstream roomsConfigFile("../configs/rooms", std::ios::binary);
    roomsConfigFile.write((char*)&numOfRooms,  sizeof(int));
    for(int i = 0; i < numOfRooms; i++){
        printRoom(rooms[i]);
        writeRoom(rooms[i], roomsConfigFile);
    }
    roomsConfigFile.close();
}

void testConfig(){
    int numOfRooms = 4;

    RoomConfig *rooms = (RoomConfig*)calloc(numOfRooms , sizeof(RoomConfig));
    memset(rooms[0].name, '\0', ROOM_NAME_SIZE);
    memset(rooms[1].name, '\0', ROOM_NAME_SIZE);
    memset(rooms[2].name, '\0', ROOM_NAME_SIZE);
    

    rooms[0].id = 1 << 0;
    rooms[0].numExtensions = 0;
    rooms[0].name[0] = 's'; rooms[0].name[1] = 'a';
    rooms[0].name[2] = 'l'; rooms[0].name[3] = 'a';
    rooms[0].minH = 5; rooms[0].maxH = 5;
    rooms[0].minW = 5; rooms[0].maxW = 5;
    rooms[0].step = 10;
    
    rooms[1].id = 1 << 1;
    rooms[1].numExtensions = 0;
    rooms[1].name[0] = 'b'; rooms[1].name[1] = 'a';
    rooms[1].name[2] = 'n'; rooms[1].name[3] = 'h';
    rooms[1].name[4] = 'e'; rooms[1].name[5] = 'i';
    rooms[1].name[6] = 'r'; rooms[1].name[7] = 'o';
    rooms[1].minH = 10; rooms[1].maxH = 10;
    rooms[1].minW = 10; rooms[1].maxW = 10;
    rooms[1].step = 10;

    rooms[2].id = 1 << 2;
    rooms[2].numExtensions = 0;
    rooms[2].name[0] = 'q'; rooms[2].name[1] = 'u';
    rooms[2].name[2] = 'a'; rooms[2].name[3] = 'r';
    rooms[2].name[4] = 't'; rooms[2].name[5] = 'o';
    rooms[2].minH = 20; rooms[2].maxH = 20;
    rooms[2].minW = 20; rooms[2].maxW = 20;
    rooms[2].step = 10;
    
    rooms[3].id = 1 << 3;
    rooms[3].numExtensions = 0;
    rooms[3].name[0] = 'c'; rooms[3].name[1] = 'o';
    rooms[3].name[2] = 'r'; rooms[3].name[3] = 'r';
    rooms[3].name[4] = 'e'; rooms[3].name[5] = 'd';
    rooms[3].name[6] = 'o'; rooms[3].name[7] = 'r';
    rooms[3].minH = 40; rooms[3].maxH = 40;
    rooms[3].minW = 40; rooms[3].maxW = 40;
    rooms[3].step = 10;

    std::ofstream roomsConfigFile("../configs/rooms", std::ios::binary);
    roomsConfigFile.write((char*)&numOfRooms,  sizeof(int));
    for(int i = 0; i < numOfRooms; i++){
        printRoom(rooms[i]);
        writeRoom(rooms[i], roomsConfigFile);
    }
    roomsConfigFile.close();
}

int main(){
    normalConfig();
    // testConfig();
}