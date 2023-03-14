#include <iostream>
#include <vector>
#include <algorithm>
#include "../lib/iter.h"
#include "../lib/generate.h"
#include "../lib/globals.h"
#include "../lib/calculator.h"
#include "../lib/cvHelper.h"


/** 
 * @brief Generate Constructor
 * @return None
*/
Generate::Generate(){
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
inline bool Generate::check_overlap(int a_left, int a_right, int a_up, int a_down, int b_left, int b_right, int b_up, int b_down){   
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
    @brief Given two squares, returns true if one square touchs another
    @param[in] a_left   left side of square A (smallest value of the x axis)
    @param[in] a_right  right side of square A (biggest value of the x axis)
    @param[in] a_up     up side of square A (smallest value of the y axis)
    @param[in] a_down   down side of square A (biggest value of the y axis)
    @param[in] b_left   left side of square B (smallest value of the x axis)
    @param[in] b_right  right side of square B (biggest value of the x axis)
    @param[in] b_up     up side of square B (smallest value of the y axis)
    @param[in] b_down   down side of square B (biggest value of the y axis)
    @return (bool) true share of a same edge (even if it is partially), false otherwise
*/
inline bool Generate::check_adjacency(int a_left, int a_right, int a_up, int a_down, int b_left, int b_right, int b_up, int b_down){  
    if((a_down == b_up || a_up == b_down) &&
        ((a_right > b_left && a_right <= b_right) ||
        (a_left < b_right && a_left >= b_left)))
            return true;   

    if((a_left == b_right || a_right == b_left) &&
        ((a_down > b_up && a_down <= b_down) ||
        (a_up < b_down && a_up >= b_up)))
            return true; 

    return false;
}


/*!
    @brief Given a vector of RoomConfig setups, iterate over every possible room sizes
    @param[in] reqSize lengh of required matrix
    @param[in] allReq required rooms ajacency, used to force room adjacency in layout, such as a master room has to have a connection with a bathroom
    @param[in] allReqCount required rooms ajacency count of how many rules are related to each room class
    @param[in] rooms vector containg all rooms informations, such as minimum and maximum sizes
    @return vector of vector of vector of layout combination. result[a][b][c] = d, a -> room size id, b -> permutation id, d -> layout points
*/
std::vector<std::vector<std::vector<int16_t>>> Generate::SizeLoop(
    const int reqSize,
    std::vector<int> allReq,
    std::vector<int> allReqCount,
    const std::vector<RoomConfig>& rooms)
    {
    
    // SizeLoopRes res;
    const int n = rooms.size();

    //Create array with the current size of every room
    std::vector<int16_t> sizeH; sizeH.reserve(n); // Height
    std::vector<int16_t> sizeW; sizeW.reserve(n); // Width
    for(int i = 0; i < n; i++){
        sizeH.push_back(rooms[i].minH);
        sizeW.push_back(rooms[i].minW);
    }
    
    for(const RoomConfig room : rooms)
        allReqCount[room.rPlannyId] -= 1;

    std::vector<int> req(allReq.size(), 0);
    for(int i = 0; i < reqSize; i++){
        for(int j = 0; j < reqSize; j++){
            if(allReqCount[i] != 0){
                allReq[i*reqSize + j] = REQ_NONE;
                allReq[j*reqSize + i] = REQ_NONE;
            }
        }
    }

    // for(int i = 0; i < allReqCount.size(); i++){
    //     if(allReqCount[i] == 0){

    //     }
    // }

    // //Create array withr RPlanny Id of each room
    // std::vector<int16_t> rIds; rIds.reserve(n);
    // for(int i = 0; i < n; i++){
    //     rIds.push_back(rooms[i].rPlannyId);
    // }

    const int NSizes = Calculator::NRoomSizes(rooms);
    std::vector<std::vector<std::vector<int16_t>>> perms; perms.reserve(NSizes);
    
    // iterate over each room size combination
    do {
        std::cout << "#########################" << std::endl;
        for(int i = 0; i < n; i++){
            std::cout << rooms[i].name << ": " << sizeW[i] << ", " << sizeH[i] << std::endl;
        }
        std::cout << "#########################" << std::endl << std::endl << std::endl;

        perms.push_back(roomPerm(n, reqSize, &sizeH[0], &sizeW[0], allReq, rooms));
        break;
    } while(Iter::nextRoomSize(rooms, &sizeH[0], &sizeW[0]));

    return perms;
}


/*!
    @brief Iterate over every possible connection between the given rooms 
    @param[in] n     number of rooms
    @param[in] NConn Number of possible connections
    @param[in] reqSize lengh of required matrix
    @param[in] sizeH Height value of each room setup
    @param[in] sizeW Width value of each room setup
    @param[in] order, specify the order of the rooms to connect
    @param[in] reqAdj required rooms ajacency, used to force room adjacency in layout, such as a master room has to have a connection with a bathroom
    @param[in] rooms vector containg all rooms informations, such as minimum and maximum sizes
    @return vector with layout points for every successful connection (n*4 int per layout)
*/
std::vector<int16_t> Generate::ConnLoop(
    const int n, 
    const int NConn, 
    const int reqSize,
    const int16_t *sizeH, 
    const int16_t *sizeW, 
    const std::vector<int>& order, 
    const std::vector<int>& reqAdj,
    const std::vector<RoomConfig>& rooms)
    {
        
    std::vector<int16_t> result; 
    std::vector<int16_t> ptsX(n * 2, 0); 
    std::vector<int16_t> ptsY(n * 2, 0);
    std::vector<int> adj(reqAdj.size()); 

    result.reserve(NConn*4*n);
    // ptsX.reserve(n * 2) ;
    // ptsY.reserve(n * 2) ;
        
    // std::cout << "perm: ";
    // for(int j = 0; j < n; j++)
    //     std::cout << order[j] << ", ";
    // std::cout << std::endl;

    for(int i = 0; i < NConn; i++){

        const int init_idx = order[0];
        ptsX[0] = 0;
        ptsY[0] = 0;
        ptsX[1] = sizeW[init_idx];
        ptsY[1] = sizeH[init_idx];
        
        int dstX = 0;
        int dstY = 0;
        int dstH = sizeH[init_idx];
        int dstW = sizeW[init_idx];
        int srcH = sizeH[init_idx];
        int srcW = sizeW[init_idx];

        std::fill(adj.begin(), adj.end(), 0);

        int sum = 0;
        bool sucess = true;
        for(int j = 1; j < n && sucess; j++){
            const int pos = (n - j - 1) * 4;
            const int srcConn = (i >> pos) & 0b11;
            const int dstConn = ((i >> (pos + 2)) & 0b11);
            if(srcConn == dstConn){
                sucess = false;
                const int diff = n - j - 1;
                sum = (1 << (diff * 4)) - 1;
                break;
            }
            
            const int j_idx = order[j];
            dstH = sizeH[j_idx];
            dstW = sizeW[j_idx];
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
                if(Generate::check_overlap(
                    ptsX[k*2], ptsX[k*2 + 1], ptsY[k*2], ptsY[k*2 + 1], 
                    ptsX[dstIndex], ptsX[dstIndex + 1], ptsY[dstIndex], ptsY[dstIndex + 1]))
                {
                    sucess = false;
                    const int diff = n - j - 1;
                    sum = (1 << (diff * 4)) - 1;
                    break;
                }

                
                if(Generate::check_adjacency(
                    ptsX[k*2], ptsX[k*2 + 1], ptsY[k*2], ptsY[k*2 + 1], 
                    ptsX[dstIndex], ptsX[dstIndex + 1], ptsY[dstIndex], ptsY[dstIndex + 1]))
                {
                    const int k_idx = order[k];
                    adj[j_idx*reqSize + k_idx] |= 1 << rooms[j_idx].id;
                    adj[k_idx*reqSize + j_idx] |= 1 << rooms[k_idx].id;
                }
            }
        }

        if(sucess){
            // result.insert(result.end(), ptsX.begin(), ptsX.end());
            // result.insert(result.end(), ptsY.begin(), ptsY.end());
            int pos = 0;
            int sucessReq = 1;
            for(int j = 0; j < reqSize; j++ && sucessReq){
                for(int k = 0; k < reqSize; k++ && sucessReq){
                    if(reqAdj[pos] == REQ_ALL)
                        sucessReq = rooms[j].familyIds & adj[pos];
                    if(reqAdj[pos] == REQ_ANY)
                        sucessReq = rooms[j].familyIds | adj[pos];

                    pos++;
                }
            }

            if(sucessReq){
                for(int j = 0; j < n; j++){
                    result.push_back(ptsX[2 * j]);
                    result.push_back(ptsY[2 * j]);
                    result.push_back(ptsX[2 * j + 1]);
                    result.push_back(ptsY[2 * j + 1]);
                }
                // result.push_back(i);
                #ifdef OPENCV_ENABLED
                CVHelper::showLayout(ptsX, ptsY);
                #endif
            }
        }
        i += sum;
    }

    return result;
}


/*!
    @brief Iterate over every room permutation
    @param[in] n number of rooms
    @param[in] reqSize lengh of required matrix
    @param[in] sizeH Height value of each room setup
    @param[in] sizeW Width value of each room setup
    @param[in] reqAdj required rooms ajacency, used to force room adjacency in layout, such as a master room has to have a connection with a bathroom
    @param[in] rooms vector containg all rooms informations, such as minimum and maximum sizes
    @return  vector of vector of layout combination. result[a][b] = c, a -> permutation id, c -> layout points
*/
std::vector<std::vector<int16_t>> Generate::roomPerm(
    const int n, 
    const int reqSize,
    const int16_t *sizeH, 
    const int16_t *sizeW, 
    const std::vector<int>& reqAdj,
    const std::vector<RoomConfig>& rooms)
    {

    std::vector<int> perm;
    for(int i = 0; i < n; i++)
        perm.push_back(i);

    // PermLoopRes res;
    const int NPerm = Calculator::Factorial(n);
    const int NConn = Calculator::NConnections(n);

    std::vector<std::vector<int16_t>> conns; 
    conns.reserve(NPerm);

    // Cycle each permutation
    do {
        // for(int i = 0; i < (int)perm.size(); i++){
        //     std::cout << perm[i] << ", ";
        // }
        // std::cout << std::endl;

        conns.push_back(ConnLoop(n, NConn, reqSize, sizeH, sizeW, perm, reqAdj, rooms));
        break;
    } while (std::next_permutation(perm.begin(), perm.end()));

    return conns;
}

// // 0--1
// // -  -
// // 2--3