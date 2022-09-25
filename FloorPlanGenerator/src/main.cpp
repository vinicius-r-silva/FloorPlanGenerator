#include <iostream>
#include <vector>
#include <algorithm>   
#include <omp.h>
// #include <sstream>    
#include <opencv2/opencv.hpp>
#include <stdlib.h>

#include "../lib/log.h"
#include "../lib/storage.h"
#include "../lib/combination.h"

void SizeLoop(const std::vector<RoomConfig> rooms);

int main( int ac, char **av){
    Storage hdd = Storage();
    // std::vector<RoomConfig> setups = hdd.getConfigs();

    // std::vector<std::vector<RoomConfig>> allCombs = getAllComb(setups, 3);
    // for(int i = 0; i < allCombs.size(); i++){
    //     for(int k = 0; k < allCombs[i].size(); k++){
    //         std::cout << allCombs[i][k].name << ",  ";
    //     }
    //     std::cout << std::endl;
    //     SizeLoop(allCombs[i]);
    //     break;
    // }
    // return 0;
}

void SizeLoop(const std::vector<RoomConfig> rooms){
    const int n = rooms.size();
    int *minH = (int*)calloc(n, sizeof(int));
    int *minW = (int*)calloc(n, sizeof(int));
    int *maxH = (int*)calloc(n, sizeof(int));
    int *maxW = (int*)calloc(n, sizeof(int));
    int *sizeH = (int*)calloc(n, sizeof(int));
    int *sizeW = (int*)calloc(n, sizeof(int));
    int *steps = (int*)calloc(n, sizeof(int));

    for(int i = 0; i < n; i++){
        minH[i] = rooms[i].minH;
        minW[i] = rooms[i].minW;
        maxH[i] = rooms[i].maxH;
        maxW[i] = rooms[i].maxW;
        sizeH[i] = rooms[i].minH;
        sizeW[i] = rooms[i].minW;
        steps[i] = rooms[i].step;
    }

    int flag = 0;
    while(flag < 2*n){
        // do some operation
        std::cout << "#########################" << std::endl;
        for(int i = 0; i < n; i++){
            std::cout << rooms[i].name << ": " << sizeH[i] << ", " << sizeW[i] << std::endl;
        }
        std::cout << "#########################" << std::endl << std::endl << std::endl;

        

        flag = 0;
        for(int i = 0; i < n; i++){
            if(sizeH[i] < maxH[i]){
                sizeH[i] += steps[i];
                if(sizeH[i] > maxH[i])
                    sizeH[i] = maxH[i];

                break;
            } else {
                sizeH[i] = minH[i];
                flag++;
            }
            
            if(sizeW[i] < maxW[i]){
                sizeW[i] += steps[i];
                if(sizeW[i] > maxW[i])
                    sizeW[i] = maxW[i];

                break;
            } else {
                sizeW[i] = minW[i];
                flag++;
            }
        }
    }

    free(minH);
    free(minW);
    free(maxH);
    free(maxW);
    free(sizeH);
    free(sizeW);
    free(steps);
}


// int factorial(int x){
//     if(x == 1)
//         return x;

//     return x*factorial(x-1);
// }

// int NConnections(int n){
//     int exponent = (n*2) - 1;

//     int res = 1;
//     for(int i = 0; i < exponent; i++){
//         res *= 4;
//     }

//     return res;
// }

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