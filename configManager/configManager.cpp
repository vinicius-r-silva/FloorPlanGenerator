#include <iostream>
#include <fstream>
#include <string.h>
#include <inttypes.h>
#include <map>
#include "../FloorPlanGenerator/lib/globals.h"
# include "csv.hpp"
#include <math.h> 
#include <filesystem>
#include <set>

// Used to get RPlanny values, such as adj constant
enum RPlannyIds {
    _LIVING_ROOM = 0,
    _MASTER_ROOM = 1,
    _KITCHEN = 2,
    _BATHROOM = 3,
    _DINING_ROOM = 4,
    _CHILD_ROOM = 5,
    _STUDY_ROOM = 6,
    _SECOND_ROOM = 7,
    _GUEST_ROOM = 8,
    _BALCONY = 9,
    _ENTRANCE = 10,
    _STORAGE = 11,
    _WALL_IN = 12,
    _EXTERNAL_AREA = 13,
    _EXTERIOR_WALL = 14,
    _FRONT_DOOR = 15,
    _INTERIOR_WALL = 16,
    _INTERIOR_DOOR = 17,
    _COUNT_RPLANNY_IDS = 18
};

template <typename T>
void printVector1D(std::vector<T> arr){
    for(T val : arr){
        std::cout << val << ", ";
    }
   std::cout <<  std::endl;
}

template <typename T>
void printVector2D(std::vector<std::vector<T>> matrix){
    for(std::vector<T> arr : matrix){
        printVector1D<T>(arr);
    }
   std::cout <<  std::endl;
}

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
    file.write((char*)&(room.familyIds), sizeof(room.familyIds));
    file.write((char*)&(room.rPlannyId), sizeof(room.rPlannyId));
    // file.write((char*)&(room.nameId), sizeof(room.nameId));
    file.write((char*)&(room.name), ROOM_NAME_SIZE * sizeof(char));
}

void printRoom(RoomConfig room){
    std::cout << room.id << " " << room.name << ": H (" << room.minH << " - " << room.maxH << 
    "), : W (" << room.minW << " - " << room.maxW << "), Ext: " << room.numExtensions << 
    ", Step: " << room.step << ", Depend: " << room.depend << ", familyIds: " << room.familyIds << 
    ", RPlannyId: " << room.rPlannyId  <<  std::endl;
    
    // ", nameId: " << room.nameId << std::endl;
}


bool setContains(std::set<int> set, int value){
    return !set.emplace(value).second;
}


template <typename T, typename V>
bool mapContains(const std::set<T, V> map, T key){
    return map.count(key) != 0;
}


/*!
    @brief Given a set of Ids from RPlanny, creates a map that links each id to the adj matrix index
    @param[in] projectPath adj matrix folder location
    @param[in] rPlannyIds  Map RPlanny Id to index in the adj matrix
    @return (map<int, int>) map Rplanny Id to index
*/
std::map<int, int> saveRplannyAdj(std::string projectPath, std::set<int> rPlannyIds){
    std::set<int> usedIds; //Ids already readed
    std::map<int, int> mapRplannyId; //result
    std::vector<int> adjValues; //map in vector form

    csv::CSVReader reader(projectPath + "/configManager/adj.csv");

    int i = 0;
    for (csv::CSVRow& row: reader) { 
        int n_cols = row.size();
        csv::CSVField field = row[0];

        int srcId = stoi(field.get<>());
        if(!setContains(rPlannyIds, srcId))
            continue;

        usedIds.insert(srcId);
        mapRplannyId.insert(std::pair<int, int>(srcId, i));
        std::vector<std::string> cols = row.get_col_names();
        
        // // for (csv::CSVField& field: row) {
        for(int j = i + 2; j < n_cols; j++){
            csv::CSVField field = row[j];
            int dstId = stoi(cols[j]);

            if(!setContains(rPlannyIds, dstId))
                continue;

            double dValue = std::stod(field.get<>());
            dValue *= 1000;
            dValue = round(dValue);
            adjValues.push_back(int(dValue));
        }

        i++;
    }

    // Error check
    if(usedIds.size() != rPlannyIds.size()){
        std::cout << "ERROR, Adj CSV does not contain all necessary informations" << std::endl;
        std::set<int>::iterator it;

        std::cout << "expected ids: " << std::endl;
        for (it=rPlannyIds.begin(); it!=rPlannyIds.end(); ++it)
            std::cout << ' ' << *it;
        std::cout << std::endl;
        
        std::cout << "found ids:" << std::endl;
        for (it=usedIds.begin(); it!=usedIds.end(); ++it)
            std::cout << ' ' << *it;

        exit(EXIT_FAILURE);
    }

    int arraySize = (int)adjValues.size();
    std::ofstream adjConfigFile(projectPath + "/configs/adj", std::ios::binary);
    adjConfigFile.write((char*)&arraySize,  sizeof(int));
    for(int i = 0; i < arraySize; i++)
        adjConfigFile.write((char*)&adjValues[i],  sizeof(adjValues[i]));
        
    adjConfigFile.close();
        
    printVector1D<int>(adjValues);
    std::cout << std::endl << std::endl;

    std::cout << "mapRplannyId: " << std::endl;
    for(auto it = mapRplannyId.cbegin(); it != mapRplannyId.cend(); ++it)
        std::cout << it->first << " " << it->second << std::endl;
        
    std::cout << std::endl;

    return mapRplannyId;
}

char asciitolower(char in) {
    if (in <= 'Z' && in >= 'A')
        return in - ('Z' - 'z');
    return in;
}

std::vector<std::string> stringToVector(std::string s, char delimiter){
    std::string tmp; 
    std::stringstream ss(s);
    std::vector<std::string> words;

    while(getline(ss, tmp, delimiter)){
        words.push_back(tmp);
    }

    // std::cout << "stringToVector. s: " << s << ", delimiter: " << delimiter << ", result:" << std::endl;
    // printVector1D<std::string>(words);
    return words;
}


std::vector<std::vector<std::string>> getReqAdjMatrix(std::string projectPath, std::map<int, int> mapRplannyId){

    csv::CSVReader reader(projectPath + "/configManager/adjreq.csv");

    
    // int n = reader.get_col_names().size() - 1;
    int n = mapRplannyId.size();
    // std::vector<std::vector<std::string>> adjReqMatrix(n, std::vector<std::string>(n, ""));
    std::vector<std::vector<std::string>>  res(n, std::vector<std::string>(n, ""));
    std::set<int> usedIds;

    int i = 0;
    for (csv::CSVRow& row: reader) { 
        int n_cols = row.size();

        csv::CSVField field = row[0];
        int srcId = stoi(field.get<>());        
        if (mapRplannyId.find(srcId) == mapRplannyId.end())
            continue;
        else
            srcId = mapRplannyId[srcId];
 
        usedIds.insert(srcId);
        std::vector<std::string> cols = row.get_col_names();

        for(int j = 1; j < n_cols; j++){
            csv::CSVField field = row[j];
            std::string val = field.get<>();

            int dstId = stoi(cols[j]);
            if (mapRplannyId.find(dstId) == mapRplannyId.end())
                continue;
            else
                dstId = mapRplannyId[dstId];

            // std::cout << "val: " << val << ", i: " << i << ", j: " << j << std::endl;
            res[srcId][dstId] = val;
        }
        i++;
    }

    if(usedIds.size() != mapRplannyId.size()){
        std::cout << std::endl << "ERROR, req Adj CSV does not contain all necessary informations" << std::endl;

        std::cout << "expected ids: " << std::endl;
        for (auto const &pair: mapRplannyId) 
            std::cout << pair.first << ", ";
        std::cout << std::endl;
        
        std::set<int>::iterator it;
        std::cout << "found ids:" << std::endl;
        for (it=usedIds.begin(); it!=usedIds.end(); ++it)
            std::cout << ' ' << *it;

        exit(EXIT_FAILURE);
    }
    return res;
}

void saveReqAdj(std::string projectPath, std::map<int, int> mapRplannyId, RoomConfig *rooms, int n){
    std::vector<std::vector<std::string>> adjReqNameMatrix = getReqAdjMatrix(projectPath, mapRplannyId);
    
    std::cout << std::endl << std::endl << "adjReqNameMatrix" << std::endl;
    printVector2D<std::string>(adjReqNameMatrix);
    std::cout << std::endl << std::endl;

    int adjReqSize = adjReqNameMatrix.size();
    std::vector<std::vector<int>> adjReqMatrix (adjReqSize, std::vector<int>(adjReqSize, 0));
    
    for(int i = 0; i < adjReqSize; i++){
        for(int j = 0; j < adjReqSize; j++){
            if(adjReqNameMatrix[i][j].compare("all") == 0)
                adjReqMatrix[i][j] = REQ_ALL;
            else if(adjReqNameMatrix[i][j].compare("any") == 0)
                adjReqMatrix[i][j] = REQ_ANY;
            else
                adjReqMatrix[i][j] = REQ_NONE;
        }
    }

    for(int i = 0; i < adjReqSize; i++){
        if(adjReqMatrix[i][i] == REQ_NONE)
            continue;

        int count = 0;
        for(int j = 0; j < n; j++){
            if(rooms[j].rPlannyId == i)
                count++;
        }

        if(count < 2)
            adjReqMatrix[i][i] = REQ_NONE;
    }
    
    std::cout << "adjReqMatrix" << std::endl;
    printVector2D<int>(adjReqMatrix);
    std::cout << std::endl << std::endl;

    int arraySize = adjReqSize * adjReqSize;
    std::ofstream adjReqConfigFile(projectPath + "/configs/reqadj", std::ios::binary);
    adjReqConfigFile.write((char*)&arraySize,  sizeof(int));
    for(int i = 0; i < adjReqSize; i++){
        for(int j = 0; j < adjReqSize; j++){
            adjReqConfigFile.write((char*)&(adjReqMatrix[i][j]),  sizeof(adjReqMatrix[i][j]));
        }
    }
        
    adjReqConfigFile.close();
}

// void extremeConfig(std::string projectPath, std::map<int, int> mapRplannyId){
//     enum ids {
//         sala = 0,
//         banheiro = 1,
//         quarto = 2,
//         corredor = 3,
//         cozinha = 4,
//         lavanderia = 5,
//         quarto2 = 6,
//         quarto3 = 7,
//         banheiro2 = 8,
//         numOfRooms = 9
//     };

//     RoomConfig *rooms = (RoomConfig*)calloc(numOfRooms , sizeof(RoomConfig));
//     for(int i = 0; i < numOfRooms; i++){
//         memset(rooms[i].name, '\0', ROOM_NAME_SIZE);
//     }

//     rooms[sala].id = 1 << sala;
//     rooms[sala].numExtensions = 2;
//     rooms[sala].name[0] = 's'; rooms[0].name[1] = 'a';
//     rooms[sala].name[2] = 'l'; rooms[0].name[3] = 'a';
//     rooms[sala].minH = 30; rooms[0].maxH = 50;
//     rooms[sala].minW = 20; rooms[0].maxW = 40;
//     rooms[sala].step = 5;
//     rooms[sala].depend = 0;
//     rooms[sala].rPlannyId = _LIVING_ROOM;
    
//     rooms[banheiro].id = 1 << banheiro;
//     rooms[banheiro].numExtensions = 0;
//     rooms[banheiro].name[0] = 'b'; rooms[1].name[1] = 'a';
//     rooms[banheiro].name[2] = 'n'; rooms[1].name[3] = 'h';
//     rooms[banheiro].name[4] = 'e'; rooms[1].name[5] = 'i';
//     rooms[banheiro].name[6] = 'r'; rooms[1].name[7] = 'o';
//     rooms[banheiro].minH = 8; rooms[1].maxH = 20;
//     rooms[banheiro].minW = 15; rooms[1].maxW = 30;
//     rooms[banheiro].step = 5;
//     rooms[banheiro].depend = 0;
//     rooms[banheiro].rPlannyId = _BATHROOM;

//     rooms[quarto].id = 1 << quarto;
//     rooms[quarto].numExtensions = 1;
//     rooms[quarto].name[0] = 'q'; rooms[2].name[1] = 'u';
//     rooms[quarto].name[2] = 'a'; rooms[2].name[3] = 'r';
//     rooms[quarto].name[4] = 't'; rooms[2].name[5] = 'o';
//     rooms[quarto].minH = 20; rooms[2].maxH = 40;
//     rooms[quarto].minW = 20; rooms[2].maxW = 40;
//     rooms[quarto].step = 5;
//     rooms[quarto].depend = 0;
//     rooms[quarto].rPlannyId = _MASTER_ROOM;
    
//     rooms[corredor].id = 1 << corredor;
//     rooms[corredor].numExtensions = 0;
//     rooms[corredor].name[0] = 'c'; rooms[3].name[1] = 'o';
//     rooms[corredor].name[2] = 'r'; rooms[3].name[3] = 'r';
//     rooms[corredor].name[4] = 'e'; rooms[3].name[5] = 'd';
//     rooms[corredor].name[6] = 'o'; rooms[3].name[7] = 'r';
//     rooms[corredor].minH = 7; rooms[3].maxH = 15;
//     rooms[corredor].minW = 7; rooms[3].maxW = 50;
//     rooms[corredor].step = 5;
//     rooms[corredor].depend = 0;
//     rooms[corredor].rPlannyId = _LIVING_ROOM;
    
//     rooms[cozinha].id = 1 << cozinha;
//     rooms[cozinha].numExtensions = 0;
//     rooms[cozinha].name[0] = 'c'; rooms[4].name[1] = 'o';
//     rooms[cozinha].name[2] = 'z'; rooms[4].name[3] = 'i';
//     rooms[cozinha].name[4] = 'n'; rooms[4].name[5] = 'h';
//     rooms[cozinha].name[6] = 'a';
//     rooms[cozinha].minH = 15; rooms[4].maxH = 25;
//     rooms[cozinha].minW = 15; rooms[4].maxW = 30;
//     rooms[cozinha].step = 5;
//     rooms[cozinha].depend = 0;
//     rooms[cozinha].rPlannyId = _KITCHEN;
    
//     rooms[lavanderia].id = 1 << lavanderia;
//     rooms[lavanderia].numExtensions = 0;
//     rooms[lavanderia].name[0] = 'l'; rooms[5].name[1] = 'a';
//     rooms[lavanderia].name[2] = 'v'; rooms[5].name[3] = 'a';
//     rooms[lavanderia].name[4] = 'n'; rooms[5].name[5] = 'd';
//     rooms[lavanderia].name[6] = 'e'; rooms[5].name[7] = 'r';
//     rooms[lavanderia].name[8] = 'i'; rooms[5].name[9] = 'a';
//     rooms[lavanderia].minH = 15; rooms[5].maxH = 25;
//     rooms[lavanderia].minW = 15; rooms[5].maxW = 30;
//     rooms[lavanderia].step = 5;
//     rooms[lavanderia].depend = 0;
//     rooms[lavanderia].rPlannyId = _KITCHEN;

//     rooms[quarto2].id = 1 << quarto2;
//     rooms[quarto2].numExtensions = 1;
//     rooms[quarto2].name[0] = 'q'; rooms[6].name[1] = 'u';
//     rooms[quarto2].name[2] = 'a'; rooms[6].name[3] = 'r';
//     rooms[quarto2].name[4] = 't'; rooms[6].name[5] = 'o';
//     rooms[quarto2].name[6] = ' '; rooms[6].name[7] = '2';
//     rooms[quarto2].minH = 20; rooms[6].maxH = 40;
//     rooms[quarto2].minW = 20; rooms[6].maxW = 40;
//     rooms[quarto2].step = 5;
//     rooms[quarto2].depend = 1 << 2;
//     rooms[quarto2].rPlannyId = _SECOND_ROOM;

//     rooms[quarto3].id = 1 << quarto3;
//     rooms[quarto3].numExtensions = 1;
//     rooms[quarto3].name[0] = 'q'; rooms[7].name[1] = 'u';
//     rooms[quarto3].name[2] = 'a'; rooms[7].name[3] = 'r';
//     rooms[quarto3].name[4] = 't'; rooms[7].name[5] = 'o';
//     rooms[quarto3].name[6] = ' '; rooms[7].name[7] = '3';
//     rooms[quarto3].minH = 20; rooms[7].maxH = 40;
//     rooms[quarto3].minW = 20; rooms[7].maxW = 40;
//     rooms[quarto3].step = 5;
//     rooms[quarto3].depend = 1 << 6;
//     rooms[quarto3].rPlannyId = _SECOND_ROOM;
    
//     rooms[banheiro2].id = 1 << banheiro2;
//     rooms[banheiro2].numExtensions = 0;
//     rooms[banheiro2].name[0] = 'b'; rooms[8].name[1] = 'a';
//     rooms[banheiro2].name[2] = 'n'; rooms[8].name[3] = 'h';
//     rooms[banheiro2].name[4] = 'e'; rooms[8].name[5] = 'i';
//     rooms[banheiro2].name[6] = 'r'; rooms[8].name[7] = 'o';
//     rooms[banheiro2].name[8] = ' '; rooms[8].name[9] = '2';
//     rooms[banheiro2].minH = 8; rooms[8].maxH = 20;
//     rooms[banheiro2].minW = 15; rooms[8].maxW = 30;
//     rooms[banheiro2].step = 5;
//     rooms[banheiro2].depend = 1 << 1;
//     rooms[banheiro2].rPlannyId = _BATHROOM;

//     std::ofstream roomsConfigFile(projectPath + "/configs/rooms", std::ios::binary);
//     roomsConfigFile.write((char*)&numOfRooms,  sizeof(int));
//     for(int i = 0; i < numOfRooms; i++){
//         printRoom(rooms[i]);
//         writeRoom(rooms[i], roomsConfigFile);
//     }
//     roomsConfigFile.close();
// }

void normalConfig(std::string projectPath){
    enum ids {
        sala = 0,
        banheiro = 1,
        quarto = 2,
        corredor = 3,
        cozinha = 4,
        lavanderia = 5,
        numOfRooms = 6
    };

    RoomConfig *rooms = (RoomConfig*)calloc(numOfRooms , sizeof(RoomConfig));
    for(int i = 0; i < numOfRooms; i++){
        memset(rooms[i].name, '\0', ROOM_NAME_SIZE);
    }

    // 0 - sala
    rooms[sala].id = 1 << sala;
    rooms[sala].numExtensions = 0; // 2
    rooms[sala].name[0] = 's'; rooms[sala].name[1] = 'a';
    rooms[sala].name[2] = 'l'; rooms[sala].name[3] = 'a';
    // rooms[sala].minH = 40; rooms[sala].maxH = 40;
    // rooms[sala].minW = 30; rooms[sala].maxW = 30;
    rooms[sala].minH = 30; rooms[sala].maxH = 50;
    rooms[sala].minW = 20; rooms[sala].maxW = 40;
    rooms[sala].step = 5; // 5
    rooms[sala].depend = 0;
    rooms[sala].rPlannyId = _LIVING_ROOM;
    
    // 1 - banheiro
    rooms[banheiro].id = 1 << banheiro;
    rooms[banheiro].numExtensions = 0;
    rooms[banheiro].name[0] = 'b'; rooms[banheiro].name[1] = 'a';
    rooms[banheiro].name[2] = 'n'; rooms[banheiro].name[3] = 'h';
    rooms[banheiro].name[4] = 'e'; rooms[banheiro].name[5] = 'i';
    rooms[banheiro].name[6] = 'r'; rooms[banheiro].name[7] = 'o';
    // rooms[banheiro].minH = 30; rooms[banheiro].maxH = 30;
    // rooms[banheiro].minW = 15; rooms[banheiro].maxW = 15;
    rooms[banheiro].minH = 8; rooms[banheiro].maxH = 20;
    rooms[banheiro].minW = 15; rooms[banheiro].maxW = 30;
    rooms[banheiro].step = 5; // 5
    rooms[banheiro].depend = 0;
    rooms[banheiro].rPlannyId = _BATHROOM;

    // 2 - quarto
    rooms[quarto].id = 1 << quarto;
    rooms[quarto].numExtensions = 0; // 1
    rooms[quarto].name[0] = 'q'; rooms[quarto].name[1] = 'u';
    rooms[quarto].name[2] = 'a'; rooms[quarto].name[3] = 'r';
    rooms[quarto].name[4] = 't'; rooms[quarto].name[5] = 'o';
    // rooms[quarto].minH = 30; rooms[quarto].maxH = 30;
    // rooms[quarto].minW = 30; rooms[quarto].maxW = 30;
    rooms[quarto].minH = 20; rooms[quarto].maxH = 40;
    rooms[quarto].minW = 20; rooms[quarto].maxW = 40;
    rooms[quarto].step = 5; // 5
    rooms[quarto].depend = 0;
    rooms[quarto].rPlannyId = _MASTER_ROOM;
    
    // 3 - corredor
    rooms[corredor].id = 1 << corredor;
    rooms[corredor].numExtensions = 0;
    rooms[corredor].name[0] = 'c'; rooms[corredor].name[1] = 'o';
    rooms[corredor].name[2] = 'r'; rooms[corredor].name[3] = 'r';
    rooms[corredor].name[4] = 'e'; rooms[corredor].name[5] = 'd';
    rooms[corredor].name[6] = 'o'; rooms[corredor].name[7] = 'r';
    // rooms[corredor].minH = 10; rooms[corredor].maxH = 10;
    // rooms[corredor].minW = 45; rooms[corredor].maxW = 45;
    rooms[corredor].minH = 7; rooms[corredor].maxH = 15;
    rooms[corredor].minW = 7; rooms[corredor].maxW = 50;
    rooms[corredor].step = 5; // 5
    rooms[corredor].depend = 0;
    rooms[corredor].rPlannyId = _LIVING_ROOM;
    
    // 4 - cozinha
    rooms[cozinha].id = 1 << cozinha;
    rooms[cozinha].numExtensions = 0;
    rooms[cozinha].name[0] = 'c'; rooms[cozinha].name[1] = 'o';
    rooms[cozinha].name[2] = 'z'; rooms[cozinha].name[3] = 'i';
    rooms[cozinha].name[4] = 'n'; rooms[cozinha].name[5] = 'h';
    rooms[cozinha].name[6] = 'a';
    // rooms[cozinha].minH = 25; rooms[cozinha].maxH = 25;
    // rooms[cozinha].minW = 20; rooms[cozinha].maxW = 20;
    rooms[cozinha].minH = 15; rooms[cozinha].maxH = 25;
    rooms[cozinha].minW = 15; rooms[cozinha].maxW = 30;
    rooms[cozinha].step = 10; // 10
    rooms[cozinha].depend = 0;
    rooms[cozinha].rPlannyId = _KITCHEN;
    
    // 5 - lavanderia
    rooms[lavanderia].id = 1 << lavanderia;
    rooms[lavanderia].numExtensions = 0;
    rooms[lavanderia].name[0] = 'l'; rooms[lavanderia].name[1] = 'a';
    rooms[lavanderia].name[2] = 'v'; rooms[lavanderia].name[3] = 'a';
    rooms[lavanderia].name[4] = 'n'; rooms[lavanderia].name[5] = 'd';
    rooms[lavanderia].name[6] = 'e'; rooms[lavanderia].name[7] = 'r';
    rooms[lavanderia].name[8] = 'i'; rooms[lavanderia].name[9] = 'a';
    // rooms[lavanderia].minH = 15; rooms[lavanderia].maxH = 15;
    // rooms[lavanderia].minW = 20; rooms[lavanderia].maxW = 20;
    rooms[lavanderia].minH = 15; rooms[lavanderia].maxH = 25;
    rooms[lavanderia].minW = 16; rooms[lavanderia].maxW = 30;
    rooms[lavanderia].step = 10; // 10
    rooms[lavanderia].depend = 0;
    rooms[lavanderia].rPlannyId = _KITCHEN;

    std::set<int> rPlannyIds;
    for(int i = 0; i < numOfRooms; i++)
        rPlannyIds.insert(rooms[i].rPlannyId);

    std::map<int, int> mapRplannyId = saveRplannyAdj(projectPath, rPlannyIds);
    for(int i = 0; i < numOfRooms; i++)
        rooms[i].rPlannyId = mapRplannyId[rooms[i].rPlannyId];
    
    std::cout << "LIVING_ROOM: " << mapRplannyId[_LIVING_ROOM] << ", BATHROOM: " << mapRplannyId[_BATHROOM] << ", MASTER_ROOM: " << mapRplannyId[_MASTER_ROOM] << ", KITCHEN: " << mapRplannyId[_KITCHEN] << std::endl;
    
    std::map<int, int> mapRIdToRooms;
    for(int i = 0; i < numOfRooms; i++){
        const int key = rooms[i].rPlannyId;
        int val = 0;
        if(mapRIdToRooms.count(key) != 0)
            val = mapRIdToRooms[key];

        val |= rooms[i].id;
        mapRIdToRooms[key] = val;
    }
    
    for(int i = 0; i < numOfRooms; i++)
        rooms[i].familyIds = mapRIdToRooms[rooms[i].rPlannyId];

    // std::unordered_map<int, std::vector<int>> mapRIdToRoomIds;
    // for(int i = 0; i < n; i++){
    //     std::vector<int> ids = mapRIdToRoomIds[rooms[i].RPlannyId];
    //     ids.push_back(rooms[i].id);
    // }

    // for(int i = 0; i < numOfRooms; i++){
    //     std::vector<int> ids = mapRIdToRoomIds[rooms[i].RPlannyId];

    //     int val = 0;
    //     for(int id : ids)
    //         val |= id;

    //     room[i].familyIds = val;
    // }

    int arrSize = numOfRooms;
    std::ofstream roomsConfigFile(projectPath + "/configs/rooms", std::ios::binary);
    roomsConfigFile.write((char*)&arrSize,  sizeof(int));
    for(int i = 0; i < numOfRooms; i++){
        printRoom(rooms[i]);
        writeRoom(rooms[i], roomsConfigFile);
    }
    roomsConfigFile.close();
 
    // std::map<const char*, int> mapNameToIdx;
    // for(int i = 0; i < numOfRooms; i++)
    //     mapNameToIdx[rooms[i].name] = i;

    saveReqAdj(projectPath, mapRplannyId, rooms, numOfRooms);
}

// void testConfig(std::string projectPath, std::map<int, int> mapRplannyId){
//     int numOfRooms = 4;

//     RoomConfig *rooms = (RoomConfig*)calloc(numOfRooms , sizeof(RoomConfig));
//     memset(rooms[0].name, '\0', ROOM_NAME_SIZE);
//     memset(rooms[1].name, '\0', ROOM_NAME_SIZE);
//     memset(rooms[2].name, '\0', ROOM_NAME_SIZE);
    

//     rooms[0].id = 1 << 0;
//     rooms[0].numExtensions = 0;
//     rooms[0].name[0] = 's'; rooms[0].name[1] = 'a';
//     rooms[0].name[2] = 'l'; rooms[0].name[3] = 'a';
//     rooms[0].minH = 5; rooms[0].maxH = 5;
//     rooms[0].minW = 5; rooms[0].maxW = 5;
//     rooms[0].step = 10;
//     rooms[0].depend = 0;
//     // rooms[0].nameId = _ID_SALA;
//     rooms[0].rPlannyId = mapRplannyId[_LIVING_ROOM];
    
//     rooms[1].id = 1 << 1;
//     rooms[1].numExtensions = 0;
//     rooms[1].name[0] = 'b'; rooms[1].name[1] = 'a';
//     rooms[1].name[2] = 'n'; rooms[1].name[3] = 'h';
//     rooms[1].name[4] = 'e'; rooms[1].name[5] = 'i';
//     rooms[1].name[6] = 'r'; rooms[1].name[7] = 'o';
//     rooms[1].minH = 10; rooms[1].maxH = 10;
//     rooms[1].minW = 10; rooms[1].maxW = 10;
//     rooms[1].step = 10;
//     rooms[1].depend = 0;
//     // rooms[1].nameId = _ID_BANHEIRO;
//     rooms[1].rPlannyId = mapRplannyId[_BATHROOM];

//     rooms[2].id = 1 << 2;
//     rooms[2].numExtensions = 0;
//     rooms[2].name[0] = 'q'; rooms[2].name[1] = 'u';
//     rooms[2].name[2] = 'a'; rooms[2].name[3] = 'r';
//     rooms[2].name[4] = 't'; rooms[2].name[5] = 'o';
//     rooms[2].minH = 20; rooms[2].maxH = 20;
//     rooms[2].minW = 20; rooms[2].maxW = 20;
//     rooms[2].step = 10;
//     rooms[2].depend = 0;
//     // rooms[2].nameId = _ID_QUARTO;
//     rooms[2].rPlannyId = mapRplannyId[_MASTER_ROOM];
    
//     rooms[3].id = 1 << 3;
//     rooms[3].numExtensions = 0;
//     rooms[3].name[0] = 'c'; rooms[3].name[1] = 'o';
//     rooms[3].name[2] = 'r'; rooms[3].name[3] = 'r';
//     rooms[3].name[4] = 'e'; rooms[3].name[5] = 'd';
//     rooms[3].name[6] = 'o'; rooms[3].name[7] = 'r';
//     rooms[3].minH = 40; rooms[3].maxH = 40;
//     rooms[3].minW = 40; rooms[3].maxW = 40;
//     rooms[3].step = 10;
//     rooms[3].depend = 0;
//     // rooms[3].nameId = _ID_SALA;
//     rooms[3].rPlannyId = mapRplannyId[_LIVING_ROOM];

//     std::ofstream roomsConfigFile(projectPath + "/configs/rooms", std::ios::binary);
//     roomsConfigFile.write((char*)&numOfRooms,  sizeof(int));
//     for(int i = 0; i < numOfRooms; i++){
//         printRoom(rooms[i]);
//         writeRoom(rooms[i], roomsConfigFile);
//     }
//     roomsConfigFile.close();
// }

int main(){
    std::string projectPath = getProjectDir();
    normalConfig(projectPath);
    // testConfig();
    // extremeConfig();
}