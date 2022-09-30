#include <iostream>
#include <vector>
#include <algorithm>   
#include <omp.h>
// #include <sstream>    
#include <stdlib.h>

#include "../lib/log.h"
#include "../lib/storage.h"
#include "../lib/iter.h"
#include "../lib/cvHelper.h"


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
        // break;
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
            std::cout << rooms[i].name << ": " << sizeW[i] << ", " << sizeH[i] << std::endl;
        }
        std::cout << "#########################" << std::endl << std::endl << std::endl;

        roomPerm(sizeH, sizeW, n);
        // break;
    } while(Iter::nextRoomSize(rooms, sizeH, sizeW));

    free(sizeH);
    free(sizeW);
}

/*!
    @brief Factorial Calculator (n!)
    @param[in] x input to calculate the factorial
    @return (int) factorial of x
*/
int Factorial(int x){
    int res = 1;
    for(; x > 1; x--)
        res *= x;

    return res;
}


/*!
    @brief Calculates the number of possible connections given the quantity of rooms
    @param[in] n input to calculate the number of connections
    @return (int) number of connections
*/
int NConnections(int n){
    int res = 1;
    for(; n > 1; n--)
        res *= 16;

    return res;
}


/*!
    @brief Given two squares, returns if there is a overleap between the two
    @param[in] a_left   left side of square A (smallest value of the x axis)
    @param[in] a_right  right side of square A (biggest value of the x axis)
    @param[in] a_up     up side of square A (smallest value of the y axis)
    @param[in] a_down   down side of square A (biggest value of the y axis)
    @param[in] b_left   left side of square B (smallest value of the x axis)
    @param[in] b_right  right side of square B (biggest value of the x axis)
    @param[in] b_up     up side of square B (smallest value of the y axis)
    @param[in] b_down   down side of square B (biggest value of the y axis)
    @return (bool) true if there is a overleap, false otherwise
*/
inline bool check_overlap(int a_left, int a_right, int a_up, int a_down, int b_left, int b_right, int b_up, int b_down){   
    if(((a_down > b_up && a_down <= b_down) ||
        (a_up  >= b_up && a_up < b_down)) &&
        ((a_right > b_left && a_right <= b_right) ||
        (a_left  >= b_left && a_left  <  b_right) ||
        (a_left  <= b_left && a_right >= b_right))){
            return true;
    }

    
    if(((b_down > a_up && b_down <= a_down) ||
        (b_up >= a_up && b_up < a_down)) &&
        ((b_right > a_left && b_right <= a_right) ||
        (b_left  >= a_left && b_left  <  a_right) ||
        (b_left  <= a_left && b_right >= a_right))){
            return true;
    }

    
    if(((a_right > b_left && a_right <= b_right) ||
        (a_left >= b_left && a_left < b_right)) &&
        ((a_down > b_up && a_down <= b_down) ||
        (a_up  >= b_up && a_up   <  b_down) ||
        (a_up  <= b_up && a_down >= b_down))){
            return true;
    }

    
    if(((b_right > a_left && b_right <= a_right) ||
        (b_left >= a_left && b_left < a_right)) &&
        ((b_down > a_up && b_down <= a_down) ||
        (b_up  >= a_up && b_up   <  a_down) ||
        (b_up  <= a_up && b_down >= a_down))){
            return true;
    }

    return false;
}


/*!
    @brief Iterate over every possible connection between the given rooms 
*/
void ConnLoop(const std::vector<int>& order, const int *sizeH, const int *sizeW, const int n, const int NConn){

    std::vector<int> ptsX(n * 2, 0); 
    std::vector<int> ptsY(n * 2, 0); 
    for(int i = 0; i < NConn; i++){
        ptsX[0] = 0;
        ptsY[0] = 0;
        ptsX[1] = sizeW[order[0]];
        ptsY[1] = sizeH[order[0]];
        
        int dstX = 0;
        int dstY = 0;
        int dstH = sizeH[order[0]];
        int dstW = sizeW[order[0]];
        int srcH = sizeH[order[0]];
        int srcW = sizeW[order[0]];

        bool sucess = true;
        for(int j = 1; j < n && sucess; j++){
            const int pos = (n - j - 1) * 4;
            const int srcConn = (i >> pos) & 0b11;
            const int dstConn = ((i >> (pos + 2)) & 0b11);
            
            dstH = sizeH[order[j]];
            dstW = sizeW[order[j]];
            if(srcConn == 1)
                dstX += srcW;
            else if(srcConn == 2)
                dstY += srcH;
            else if(srcConn == 3){
                dstX += srcW;
                dstY += srcH;
            }

            if(dstConn == 1)
                dstX -= dstW;
            else if(dstConn == 2)
                dstY -= dstH;
            else if(dstConn == 3){
                dstX -= dstW;
                dstY -= dstH;
            }

            const int dstIndex = j*2;
            ptsX[dstIndex] = dstX; ptsY[dstIndex] = dstY;
            ptsX[dstIndex + 1] = dstX + dstW; ptsY[dstIndex + 1] = dstY + dstH;
            
            dstX = ptsX[dstIndex];
            dstY = ptsY[dstIndex];
            srcH = dstH;
            srcW = dstW;
            
            for(int k = 0; k < j; k++){
                if(check_overlap(ptsX[k*2], ptsX[k*2 + 1], ptsY[k*2], ptsY[k*2 + 1], ptsX[dstIndex], ptsX[dstIndex + 1], ptsY[dstIndex], ptsY[dstIndex + 1])){
                    sucess = false;
                    break;
                }
            }
        }

        if(sucess)
            CVHelper::showLayout(ptsX, ptsY, n);
    }
}


void roomPerm(const int *sizeH, const int *sizeW, const int n){
    std::vector<int> perm;
    for(int i = 0; i < n; i++)
        perm.push_back(i);

    // const int NPerm = Factorial(n);
    const int NConn = NConnections(n);

    // std::vector<std::vector<std::vector<int>>> allPtX(NPerm * NConn); 
    // std::vector<std::vector<std::vector<int>>> allPtY(NPerm * NConn); 
    // std::vector<std::vector<short>> allPtY(NPerm * NConn, std::vector<short> (n * 4, 0)); 

    // std::vector<std::vector<short>> allPtX(NPerm * NConn, std::vector<short> (n * 4, 0)); 
    // std::vector<std::vector<short>> allPtY(NPerm * NConn, std::vector<short> (n * 4, 0)); 

    // Cycle each permutation
    int i = 0;
    do {
        ConnLoop(perm, sizeH, sizeW, n, NConn);
        i += 1;
        // break;
    } while (std::next_permutation(perm.begin(), perm.end()));

    // for(int i = 0; i < NPerm; i++){
    //     for(int j = 0; j < NConn; j++){
    //         CVHelper::showLayout(allPtX[i][j], allPtY[i][j], n);
    //     }
    // }
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