#include <iostream>
#include <vector>
#include <algorithm>   
#include <omp.h>
// #include <sstream>    
#include <opencv2/opencv.hpp>
#include <stdlib.h>

#include "../lib/log.h"
#include "../lib/storage.h"
#include "../lib/iter.h"


/*!
    @brief Given a vector of RoomConfig setups, iterate over every possible room sizes
    @param[in] rooms vector containg all rooms informations, such as minimum and maximum sizes
    @return None
*/
void SizeLoop(const std::vector<RoomConfig> rooms);


void roomPerm(const int *sizeH, const int *sizeW, const int n);


/*!
    @brief Main Function
    @return if there are no erros returns 0 
*/
int main(){
    Storage hdd = Storage();
    std::vector<RoomConfig> setups = hdd.getConfigs();

    std::vector<std::vector<RoomConfig>> allCombs = Iter::getAllComb(setups, 3);
    for(std::size_t i = 0; i < allCombs.size(); i++){
        for(std::size_t k = 0; k < allCombs[i].size(); k++){
            std::cout << allCombs[i][k].name << ",  ";
        }
        std::cout << std::endl;
        SizeLoop(allCombs[i]);
        break;
    }
    return 0;
}


/*!
    @brief Given a vector of RoomConfig setups, iterate over every possible room sizes
    @param[in] rooms vector containg all rooms informations, such as minimum and maximum sizes
    @return None
*/
void SizeLoop(const std::vector<RoomConfig> rooms){
    const int n = rooms.size();

    //Create array with the current size of every room
    int *sizeH = (int*)calloc(n, sizeof(int)); //Height
    int *sizeW = (int*)calloc(n, sizeof(int)); //Width
    for(int i = 0; i < n; i++){
        sizeH[i] = rooms[i].minH;
        sizeW[i] = rooms[i].minW;
    }

    //Main loop
    do {
        std::cout << "#########################" << std::endl;
        for(int i = 0; i < n; i++){
            std::cout << rooms[i].name << ": " << sizeH[i] << ", " << sizeW[i] << std::endl;
        }
        std::cout << "#########################" << std::endl << std::endl << std::endl;

        roomPerm(sizeH, sizeW, n);
        break;
    } while(Iter::nextRoomSize(rooms, sizeH, sizeW));

    free(sizeH);
    free(sizeW);
}


// int factorial(int x){
//     int res = 1;
//     for(; x > 1; x--)
//         res *= x;

//     return res;
// }

int NConnections(int n){
    int res = 1;
    for(; n > 1; n--)
        res *= 16;

    return res;
}


// int hibit(unsigned int n) {
//     n |= (n >>  1);
//     n |= (n >>  2);
//     n |= (n >>  4);
//     n |= (n >>  8);
//     n |= (n >> 16);
//     return n - (n >> 1);

// colocar return (n - (n >> 1)) / 4 pra descobrir quem mudou
// }

void roomPerm(const int *sizeH, const int *sizeW, const int n){
    std::vector<int> perm;
    for(int i = 0; i < n; i++)
        perm.push_back(i);

    const int NConn = NConnections(n);

    std::vector<std::vector<short>> allPtX(NConn, std::vector<short> (n * 4, 0)); 
    std::vector<std::vector<short>> allPtY(NConn, std::vector<short> (n * 4, 0)); 
    // Cycle each permutation
    int i = 0;
    // int changedIdx = 0;
    do {
        for(i = 0; i < n; i++){
            std::cout << perm[i] << ",  ";
        }
        std::cout << std::endl;


        for(i = 0; i < NConn; i++){
            // std::vector<short> ptsX = allPtX[i];
            // std::vector<short> ptsY = allPtY[i];
            std::cout << std::endl << "\t" << i << std::endl;
            allPtX[i][1] = sizeW[0];
            allPtY[i][1] = 0;
            allPtX[i][2] = 0;
            allPtY[i][2] = sizeH[0];
            allPtX[i][3] = sizeW[0];
            allPtY[i][3] = sizeH[0];
            
            int dstX = 0;
            int dstY = 0;
            for(int j = 1; j < n; j++){
                const int pos = (j - 1) * 4;
                const int srcConn = (i >> pos) & 0b11;
                const int dstConn = ((i >> (pos + 2)) & 0b11);
            
                const int dstH = sizeH[j];
                const int dstW = sizeW[j];

                if(dstConn == 1)
                    dstX -= dstW;
                else if(dstConn == 2)
                    dstY -= dstH;
                else if(dstConn == 3){
                    dstX -= dstW;
                    dstY -= dstH;
                }

                const int dstIndex = j*4;
                allPtX[i][dstIndex] = dstX; allPtY[i][dstIndex] = dstY;
                allPtX[i][dstIndex + 1] = dstX + dstW; allPtY[i][dstIndex + 1] = dstY;
                allPtX[i][dstIndex + 2] = dstX       ; allPtY[i][dstIndex + 2] = dstY + dstH;
                allPtX[i][dstIndex + 3] = dstX + dstW; allPtY[i][dstIndex + 3] = dstY + dstH;     
                
                dstX = allPtX[i][dstIndex + srcConn];
                dstY = allPtY[i][dstIndex + srcConn];
            }
        }
        // std::cout << std::endl;
        // std::cout << std::endl;
        // for(i = 0; i < NConn; i++){
        //     std::cout << "\t" << i << std::endl;
        //     for(int j = 0; j < n; j++){
        //         std::cout << "\t\t" << j << std::endl;
        //         for(int k = 0; k < 4; k++){
        //             std::cout << "\t\t\t" << k << ": (" << allPtX[i][j*4 + k]  << ", " << allPtY[i][j*4 + k] << ")" << std::endl;
        //         }
        //         std::cout << std::endl;
        //     }
        // }

        break;
    } while (std::next_permutation(perm.begin(), perm.end()));
}

// // 0--1
// // -  -
// // 2--3

// int main( int ac, char **av){
//     char n = 3;
//     std::vector<char> perm;
//     for(char i = 0; i < n; i++)
//         perm.push_back(i);

//     signed char *alturas = (signed char*)calloc(n, sizeof(signed char));
//     signed char *larguras = (signed char*)calloc(n, sizeof(signed char));
//     for(int i = 0; i < n; i++){
//         alturas[i] = 10*i;
//         larguras[i] = 5*i;
//     }
 
//     // char *ConnSndr = (char*)calloc(n, sizeof(char));
//     // char *ConnRcvr = (char*)calloc(n, sizeof(char));

//     // short *ptX = (short*)calloc(n * 4, sizeof(short));
//     // short *ptY = (short*)calloc(n * 4, sizeof(short));

//     int Nperm = factorial(n);
//     int NConn = NConnections(n);
//     std::vector<std::vector<short>> allPtX(Nperm * NConn, std::vector<short> (n * 4, 0)); 
//     std::vector<std::vector<short>> allPtY(Nperm * NConn, std::vector<short> (n * 4, 0)); 

//     std::cout << "NPerm: " << Nperm << ", nConn: " << NConn << ", Nperm * NConn: " << Nperm * NConn << std::endl;

//     // Cycle each permutation
//     int i = 0;
//     do {
//         short *ptX = allPtX[i * NConn + ]
//         short *ptY = allPtX[i * NConn + ]

//         char currRoom = perm[0];
//         char prevRoom = currRoom;
//         ptX[1] = larguras[currRoom]; ptY[2] = alturas[currRoom];
//         ptX[3] = larguras[currRoom];
        
        
        
        
        
//         tY[3] = alturas[currRoom];

//         //Cycle each environment
//         for(int i = 1; i < n - 1; i++){
//             char currRoom = perm[i];
//             int srcIndex = prevRoom * 4;
//             int dstIndex = currRoom * 4;
            
//             //Cycle each src connection
//             for(int j = 0; j < 4; j++){
//                 signed char dstX = ptX[srcIndex + j];
//                 signed char dstY = ptY[srcIndex + j];

//                 //Cycle each dst connection
//                 for(int k = 0; k < 4; k++){
//                     signed char dstA = alturas[k];
//                     signed char dstL = larguras[k];
//                     if(k == 1)
//                         dstX -= dstL;
//                     else if(k == 2)
//                         dstY -= dstA;
//                     else if(k == 3){
//                         dstX -= dstL;
//                         dstY -= dstA;
//                     }

//                     ptX[dstIndex] = dstX; ptY[dstIndex] = dstY;
//                     ptX[dstIndex + 1] = dstX + dstL; ptY[dstIndex + 1] = dstY;
//                     ptX[dstIndex + 2] = dstX       ; ptY[dstIndex + 2] = dstY + dstA;
//                     ptX[dstIndex + 3] = dstX + dstL; ptY[dstIndex + 3] = dstY + dstA;
//                 }
//             }
//             char prevRoom = currRoom;
//         }
//         i++;
//     }
//     while (std::next_permutation(perm.begin(), perm.end()));
//     std::cout << "i: " << i << std::endl;

// }