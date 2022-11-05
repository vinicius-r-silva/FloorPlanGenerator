#include <iostream>
#include <fstream>
#include <string.h>
#include <inttypes.h>
#include <map>
#include "../FloorPlanGenerator/lib/globals.h"
# include "csv.hpp"
#include <math.h> 
#include <filesystem>

enum RPlannyIds {
    LIVING_ROOM = 0,
    MASTER_ROOM = 1,
    KITCHEN = 2,
    BATHROOM = 3,
    DINING_ROOM = 4,
    CHILD_ROOM = 5,
    STUDY_ROOM = 6,
    SECOND_ROOM = 7,
    GUEST_ROOM = 8,
    BALCONY = 9,
    ENTRANCE = 10,
    STORAGE = 11,
    WALL_IN = 12,
    EXTERNAL_AREA = 13,
    EXTERIOR_WALL = 14,
    FRONT_DOOR = 15,
    INTERIOR_WALL = 16,
    INTERIOR_DOOR = 17
};


// returns the current executable directory until the first appearence of the folder "FloorPlanGenerator"
std::string getProjectDir(){
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

void writeRoom(RoomConfig room, std::ofstream& file){
    file.write((char*)&(room.id), sizeof(room.id));
    file.write((char*)&(room.step), sizeof(room.step));
    file.write((char*)&(room.numExtensions), sizeof(room.numExtensions));
    file.write((char*)&(room.minH), sizeof(room.minH));
    file.write((char*)&(room.maxH), sizeof(room.maxH));
    file.write((char*)&(room.minW), sizeof(room.minW));
    file.write((char*)&(room.maxW), sizeof(room.maxW));
    file.write((char*)&(room.depend), sizeof(room.depend));
    file.write((char*)&(room.rPlannyId), sizeof(room.rPlannyId));
    file.write((char*)&(room.name), ROOM_NAME_SIZE * sizeof(char));
}

void printRoom(RoomConfig room){
    std::cout << room.id << " " << room.name << ": H (" << room.minH << " - " << room.maxH << 
    "), : W (" << room.minW << " - " << room.maxW << "), Ext: " << room.numExtensions << 
    ", Step: " << room.step << ", Depend: " << room.depend << ", RPlannyId: " << room.rPlannyId << std::endl;
}

void extremeConfig(std::string projectPath, std::map<int, int> mapRplannyId){
    int numOfRooms = 9;

    RoomConfig *rooms = (RoomConfig*)calloc(numOfRooms , sizeof(RoomConfig));
    memset(rooms[0].name, '\0', ROOM_NAME_SIZE);
    memset(rooms[1].name, '\0', ROOM_NAME_SIZE);
    memset(rooms[2].name, '\0', ROOM_NAME_SIZE);
    memset(rooms[3].name, '\0', ROOM_NAME_SIZE);
    memset(rooms[4].name, '\0', ROOM_NAME_SIZE);
    memset(rooms[5].name, '\0', ROOM_NAME_SIZE);
    memset(rooms[6].name, '\0', ROOM_NAME_SIZE);
    memset(rooms[7].name, '\0', ROOM_NAME_SIZE);
    memset(rooms[8].name, '\0', ROOM_NAME_SIZE);
    
    rooms[0].id = 1 << 0;
    rooms[0].numExtensions = 2;
    rooms[0].name[0] = 's'; rooms[0].name[1] = 'a';
    rooms[0].name[2] = 'l'; rooms[0].name[3] = 'a';
    rooms[0].minH = 30; rooms[0].maxH = 50;
    rooms[0].minW = 20; rooms[0].maxW = 40;
    rooms[0].step = 5;
    rooms[0].depend = 0;
    rooms[0].rPlannyId = mapRplannyId[LIVING_ROOM];

    
    rooms[1].id = 1 << 1;
    rooms[1].numExtensions = 0;
    rooms[1].name[0] = 'b'; rooms[1].name[1] = 'a';
    rooms[1].name[2] = 'n'; rooms[1].name[3] = 'h';
    rooms[1].name[4] = 'e'; rooms[1].name[5] = 'i';
    rooms[1].name[6] = 'r'; rooms[1].name[7] = 'o';
    rooms[1].minH = 8; rooms[1].maxH = 20;
    rooms[1].minW = 15; rooms[1].maxW = 30;
    rooms[1].step = 5;
    rooms[1].depend = 0;
    rooms[0].rPlannyId = mapRplannyId[BATHROOM];

    rooms[2].id = 1 << 2;
    rooms[2].numExtensions = 1;
    rooms[2].name[0] = 'q'; rooms[2].name[1] = 'u';
    rooms[2].name[2] = 'a'; rooms[2].name[3] = 'r';
    rooms[2].name[4] = 't'; rooms[2].name[5] = 'o';
    rooms[2].minH = 20; rooms[2].maxH = 40;
    rooms[2].minW = 20; rooms[2].maxW = 40;
    rooms[2].step = 5;
    rooms[2].depend = 0;
    rooms[0].rPlannyId = mapRplannyId[MASTER_ROOM];
    
    rooms[3].id = 1 << 3;
    rooms[3].numExtensions = 0;
    rooms[3].name[0] = 'c'; rooms[3].name[1] = 'o';
    rooms[3].name[2] = 'r'; rooms[3].name[3] = 'r';
    rooms[3].name[4] = 'e'; rooms[3].name[5] = 'd';
    rooms[3].name[6] = 'o'; rooms[3].name[7] = 'r';
    rooms[3].minH = 7; rooms[3].maxH = 15;
    rooms[3].minW = 7; rooms[3].maxW = 50;
    rooms[3].step = 5;
    rooms[3].depend = 0;
    rooms[0].rPlannyId = mapRplannyId[LIVING_ROOM];
    
    rooms[4].id = 1 << 4;
    rooms[4].numExtensions = 0;
    rooms[4].name[0] = 'c'; rooms[4].name[1] = 'o';
    rooms[4].name[2] = 'z'; rooms[4].name[3] = 'i';
    rooms[4].name[4] = 'n'; rooms[4].name[5] = 'h';
    rooms[4].name[6] = 'a';
    rooms[4].minH = 15; rooms[4].maxH = 25;
    rooms[4].minW = 15; rooms[4].maxW = 30;
    rooms[4].step = 5;
    rooms[4].depend = 0;
    rooms[0].rPlannyId = mapRplannyId[KITCHEN];
    
    rooms[5].id = 1 << 5;
    rooms[5].numExtensions = 0;
    rooms[5].name[0] = 'l'; rooms[5].name[1] = 'a';
    rooms[5].name[2] = 'v'; rooms[5].name[3] = 'a';
    rooms[5].name[4] = 'n'; rooms[5].name[5] = 'd';
    rooms[5].name[6] = 'e'; rooms[5].name[7] = 'r';
    rooms[5].name[8] = 'i'; rooms[5].name[9] = 'a';
    rooms[5].minH = 15; rooms[5].maxH = 25;
    rooms[5].minW = 15; rooms[5].maxW = 30;
    rooms[5].step = 5;
    rooms[5].depend = 0;
    rooms[0].rPlannyId = mapRplannyId[KITCHEN];

    rooms[6].id = 1 << 6;
    rooms[6].numExtensions = 1;
    rooms[6].name[0] = 'q'; rooms[6].name[1] = 'u';
    rooms[6].name[2] = 'a'; rooms[6].name[3] = 'r';
    rooms[6].name[4] = 't'; rooms[6].name[5] = 'o';
    rooms[6].name[6] = ' '; rooms[6].name[7] = '2';
    rooms[6].minH = 20; rooms[6].maxH = 40;
    rooms[6].minW = 20; rooms[6].maxW = 40;
    rooms[6].step = 5;
    rooms[6].depend = 1 << 2;
    rooms[0].rPlannyId = mapRplannyId[SECOND_ROOM];

    rooms[7].id = 1 << 7;
    rooms[7].numExtensions = 1;
    rooms[7].name[0] = 'q'; rooms[7].name[1] = 'u';
    rooms[7].name[2] = 'a'; rooms[7].name[3] = 'r';
    rooms[7].name[4] = 't'; rooms[7].name[5] = 'o';
    rooms[7].name[6] = ' '; rooms[7].name[7] = '3';
    rooms[7].minH = 20; rooms[7].maxH = 40;
    rooms[7].minW = 20; rooms[7].maxW = 40;
    rooms[7].step = 5;
    rooms[7].depend = 1 << 6;
    rooms[0].rPlannyId = mapRplannyId[SECOND_ROOM];
    
    rooms[8].id = 1 << 8;
    rooms[8].numExtensions = 0;
    rooms[8].name[0] = 'b'; rooms[8].name[1] = 'a';
    rooms[8].name[2] = 'n'; rooms[8].name[3] = 'h';
    rooms[8].name[4] = 'e'; rooms[8].name[5] = 'i';
    rooms[8].name[6] = 'r'; rooms[8].name[7] = 'o';
    rooms[8].name[8] = ' '; rooms[8].name[9] = '2';
    rooms[8].minH = 8; rooms[8].maxH = 20;
    rooms[8].minW = 15; rooms[8].maxW = 30;
    rooms[8].step = 5;
    rooms[8].depend = 1 << 1;
    rooms[0].rPlannyId = mapRplannyId[BATHROOM];

    std::ofstream roomsConfigFile(projectPath + "/configs/rooms", std::ios::binary);
    roomsConfigFile.write((char*)&numOfRooms,  sizeof(int));
    for(int i = 0; i < numOfRooms; i++){
        printRoom(rooms[i]);
        writeRoom(rooms[i], roomsConfigFile);
    }
    roomsConfigFile.close();
}

void normalConfig(std::string projectPath, std::map<int, int> mapRplannyId){
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
    rooms[0].step = 100;
    rooms[0].depend = 0;
    rooms[0].rPlannyId = mapRplannyId[LIVING_ROOM];
    
    rooms[1].id = 1 << 1;
    rooms[1].numExtensions = 0;
    rooms[1].name[0] = 'b'; rooms[1].name[1] = 'a';
    rooms[1].name[2] = 'n'; rooms[1].name[3] = 'h';
    rooms[1].name[4] = 'e'; rooms[1].name[5] = 'i';
    rooms[1].name[6] = 'r'; rooms[1].name[7] = 'o';
    rooms[1].minH = 8; rooms[1].maxH = 20;
    rooms[1].minW = 15; rooms[1].maxW = 30;
    rooms[1].step = 100;
    rooms[1].depend = 0;
    rooms[1].rPlannyId = mapRplannyId[BATHROOM];

    rooms[2].id = 1 << 2;
    rooms[2].numExtensions = 1;
    rooms[2].name[0] = 'q'; rooms[2].name[1] = 'u';
    rooms[2].name[2] = 'a'; rooms[2].name[3] = 'r';
    rooms[2].name[4] = 't'; rooms[2].name[5] = 'o';
    rooms[2].minH = 20; rooms[2].maxH = 40;
    rooms[2].minW = 20; rooms[2].maxW = 40;
    rooms[2].step = 100;
    rooms[2].depend = 0;
    rooms[2].rPlannyId = mapRplannyId[MASTER_ROOM];
    
    rooms[3].id = 1 << 3;
    rooms[3].numExtensions = 0;
    rooms[3].name[0] = 'c'; rooms[3].name[1] = 'o';
    rooms[3].name[2] = 'r'; rooms[3].name[3] = 'r';
    rooms[3].name[4] = 'e'; rooms[3].name[5] = 'd';
    rooms[3].name[6] = 'o'; rooms[3].name[7] = 'r';
    rooms[3].minH = 7; rooms[3].maxH = 15;
    rooms[3].minW = 7; rooms[3].maxW = 50;
    rooms[3].step = 100;
    rooms[3].depend = 0;
    rooms[3].rPlannyId = mapRplannyId[LIVING_ROOM];
    
    rooms[4].id = 1 << 4;
    rooms[4].numExtensions = 0;
    rooms[4].name[0] = 'c'; rooms[4].name[1] = 'o';
    rooms[4].name[2] = 'z'; rooms[4].name[3] = 'i';
    rooms[4].name[4] = 'n'; rooms[4].name[5] = 'h';
    rooms[4].name[6] = 'a';
    rooms[4].minH = 15; rooms[4].maxH = 25;
    rooms[4].minW = 15; rooms[4].maxW = 30;
    rooms[4].step = 100;
    rooms[4].depend = 0;
    rooms[4].rPlannyId = mapRplannyId[KITCHEN];
    
    rooms[5].id = 1 << 5;
    rooms[5].numExtensions = 0;
    rooms[5].name[0] = 'l'; rooms[5].name[1] = 'a';
    rooms[5].name[2] = 'v'; rooms[5].name[3] = 'a';
    rooms[5].name[4] = 'n'; rooms[5].name[5] = 'd';
    rooms[5].name[6] = 'e'; rooms[5].name[7] = 'r';
    rooms[5].name[8] = 'i'; rooms[5].name[9] = 'a';
    rooms[5].minH = 15; rooms[5].maxH = 25;
    rooms[5].minW = 16; rooms[5].maxW = 30;
    rooms[5].step = 100;
    rooms[5].depend = 0;
    rooms[5].rPlannyId = mapRplannyId[KITCHEN];


    std::cout << "LIVING_ROOM: " << mapRplannyId[LIVING_ROOM] << ", BATHROOM: " << mapRplannyId[BATHROOM] << ", MASTER_ROOM: " << mapRplannyId[MASTER_ROOM] << ", KITCHEN: " << mapRplannyId[KITCHEN] << std::endl;

    std::ofstream roomsConfigFile(projectPath + "/configs/rooms", std::ios::binary);
    roomsConfigFile.write((char*)&numOfRooms,  sizeof(int));
    for(int i = 0; i < numOfRooms; i++){
        printRoom(rooms[i]);
        writeRoom(rooms[i], roomsConfigFile);
    }
    roomsConfigFile.close();
}

void testConfig(std::string projectPath, std::map<int, int> mapRplannyId){
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
    rooms[0].depend = 0;
    rooms[0].rPlannyId = mapRplannyId[LIVING_ROOM];
    
    rooms[1].id = 1 << 1;
    rooms[1].numExtensions = 0;
    rooms[1].name[0] = 'b'; rooms[1].name[1] = 'a';
    rooms[1].name[2] = 'n'; rooms[1].name[3] = 'h';
    rooms[1].name[4] = 'e'; rooms[1].name[5] = 'i';
    rooms[1].name[6] = 'r'; rooms[1].name[7] = 'o';
    rooms[1].minH = 10; rooms[1].maxH = 10;
    rooms[1].minW = 10; rooms[1].maxW = 10;
    rooms[1].step = 10;
    rooms[1].depend = 0;
    rooms[0].rPlannyId = mapRplannyId[BATHROOM];

    rooms[2].id = 1 << 2;
    rooms[2].numExtensions = 0;
    rooms[2].name[0] = 'q'; rooms[2].name[1] = 'u';
    rooms[2].name[2] = 'a'; rooms[2].name[3] = 'r';
    rooms[2].name[4] = 't'; rooms[2].name[5] = 'o';
    rooms[2].minH = 20; rooms[2].maxH = 20;
    rooms[2].minW = 20; rooms[2].maxW = 20;
    rooms[2].step = 10;
    rooms[2].depend = 0;
    rooms[0].rPlannyId = mapRplannyId[MASTER_ROOM];
    
    rooms[3].id = 1 << 3;
    rooms[3].numExtensions = 0;
    rooms[3].name[0] = 'c'; rooms[3].name[1] = 'o';
    rooms[3].name[2] = 'r'; rooms[3].name[3] = 'r';
    rooms[3].name[4] = 'e'; rooms[3].name[5] = 'd';
    rooms[3].name[6] = 'o'; rooms[3].name[7] = 'r';
    rooms[3].minH = 40; rooms[3].maxH = 40;
    rooms[3].minW = 40; rooms[3].maxW = 40;
    rooms[3].step = 10;
    rooms[3].depend = 0;
    rooms[0].rPlannyId = mapRplannyId[LIVING_ROOM];

    std::ofstream roomsConfigFile(projectPath + "/configs/rooms", std::ios::binary);
    roomsConfigFile.write((char*)&numOfRooms,  sizeof(int));
    for(int i = 0; i < numOfRooms; i++){
        printRoom(rooms[i]);
        writeRoom(rooms[i], roomsConfigFile);
    }
    roomsConfigFile.close();
}

std::map<int, int> saveRplannyAdj(std::string projectPath){
    std::map<int, int> mapRplannyId;
    std::vector<int> adjValues;

    csv::CSVReader reader(projectPath + "/configManager/adj.csv");

    int i = 0;
    for (csv::CSVRow& row: reader) { 
        int n_cols = row.size();
        csv::CSVField field = row[0];

        int srcId = stoi(field.get<>());
        mapRplannyId.insert(std::pair<int, int>(srcId, i));
        
        // // for (csv::CSVField& field: row) {
        for(int j = i + 2; j < n_cols; j++){
            csv::CSVField field = row[j];

            double dValue = std::stod(field.get<>());
            dValue *= 1000;
            dValue = round(dValue);

            adjValues.push_back(int(dValue));
        }

        i++;
    }

    int arraySize = (int)adjValues.size();
    std::ofstream adjConfigFile(projectPath + "/configs/adj", std::ios::binary);
    adjConfigFile.write((char*)&arraySize,  sizeof(arraySize));
    for(int i = 0; i < arraySize; i++){
        adjConfigFile.write((char*)&adjValues[i],  sizeof(adjValues[i]));
    }
    adjConfigFile.close();

    for(int val : adjValues){
        std::cout << val << ", ";
    }
    std::cout << "\n";

    // for(auto it = mapRplannyId.cbegin(); it != mapRplannyId.cend(); ++it)
    // {
    //     std::cout << it->first << " " << it->second << "\n";
    // }

    return mapRplannyId;
}

int main(){
    std::string projectPath = getProjectDir();
    std::map<int, int> mapRplannyId = saveRplannyAdj(projectPath);
    normalConfig(projectPath, mapRplannyId);
    // testConfig();
    // extremeConfig();
}