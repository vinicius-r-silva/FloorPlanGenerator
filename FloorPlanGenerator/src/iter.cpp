#include <iostream>
#include <vector>
#include <algorithm>
#include "../lib/iter.h"
#include "../lib/globals.h"
#include "../lib/calculator.h"
#include "../lib/cvHelper.h"


/** 
 * @brief Iter Constructor
 * @return None
*/
Iter::Iter(){
}

/*!
    @brief Get the next combination of k elements in a vector of size n
    @details https://stackoverflow.com/questions/5095407/all-combinations-of-k-elements-out-of-n
    @param[in] first begin of the vector
    @param[in] k current position 
    @param[in] last end of the vector
    @return True if there is a new combination, false otherwise
*/
template <typename Iterator>
inline bool Iter::next_combination(const Iterator first, Iterator k, const Iterator last){
    /* Credits: Thomas Draper */
    if ((first == last) || (first == k) || (last == k))
        return false;
    Iterator itr1 = first;
    Iterator itr2 = last;
    ++itr1;
    if (last == itr1)
        return false;
    itr1 = last;
    --itr1;
    itr1 = k;
    --itr2;
    while (first != itr1)
    {
        if (*--itr1 < *itr2)
        {
            Iterator j = k;
            while (!(*itr1 < *j)) ++j;
            std::iter_swap(itr1,j);
            ++itr1;
            ++j;
            itr2 = k;
            std::rotate(itr1,j,last);
            while (last != j)
            {
                ++j;
                ++itr2;
            }
            std::rotate(k,itr2,last);
            return true;
        }
    }
    std::rotate(first,k,last);
    return false;
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
inline bool Iter::check_overlap(int a_left, int a_right, int a_up, int a_down, int b_left, int b_right, int b_up, int b_down){   
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
    @brief Get all possible combinations of k elements in a vector of size n
    @details https://stackoverflow.com/questions/5095407/all-combinations-of-k-elements-out-of-n
    @param[in] setups vector containg all elements
    @param[in] k size of the combinations
    @return (vector of vector of RoomConfig) return a vector with all possible combinations where wich combination is a vector of RoomConfig 
*/
std::vector<std::vector<RoomConfig>> Iter::getAllComb(std::vector<RoomConfig> setups, int k){
    std::vector<std::vector<RoomConfig>> result = std::vector<std::vector<RoomConfig>>();

    int n = setups.size();
    std::vector<int> setupIdx;
    for (int i = 0; i < n; setupIdx.push_back(i++));

    do {
        std::vector<RoomConfig> comb = std::vector<RoomConfig>();
        for (int i = 0; i < k; ++i){
            comb.push_back(setups[setupIdx[i]]);
        }
        result.push_back(comb);
    } while(Iter::next_combination(setupIdx.begin(),setupIdx.begin() + k, setupIdx.end()));
    return result;
}


/*!
    @brief Calculate a new room's width and height
    @param[in] rooms vector containg all rooms informations, such as minimum and maximum sizes
    @param[out] sizeH rooms Height size
    @param[out] sizeW rooms Width size
    @return True if there is a next room size iteration, false otherwise
*/
bool Iter::nextRoomSize(std::vector<RoomConfig> rooms, int *sizeH, int *sizeW){
    int n = rooms.size();
    int flag = 0;
    for(int i = 0; i < n; i++){
        if(sizeH[i] < rooms[i].maxH){
            sizeH[i] += rooms[i].step;
            if(sizeH[i] > rooms[i].maxH)
                sizeH[i] = rooms[i].maxH;

            break;
        } else {
            sizeH[i] = rooms[i].minH;
            flag++;
        }
        
        if(sizeW[i] < rooms[i].maxW){
            sizeW[i] += rooms[i].step;
            if(sizeW[i] > rooms[i].maxW)
                sizeW[i] = rooms[i].maxW;

            break;
        } else {
            sizeW[i] = rooms[i].minW;
            flag++;
        }
    }

    return flag < 2*n;
}




/*!
    @brief Given a vector of RoomConfig setups, iterate over every possible room sizes
    @param[in] rooms vector containg all rooms informations, such as minimum and maximum sizes
    @return None
*/
SizeLoopRes Iter::SizeLoop(const std::vector<RoomConfig>& rooms){
    SizeLoopRes res;
    const int n = rooms.size();

    std::vector<int> roomsId;
    roomsId.reserve(n);

    //Create array with the current size of every room
    int *sizeH = (int*)calloc(n, sizeof(int)); //Height
    int *sizeW = (int*)calloc(n, sizeof(int)); //Width
    for(int i = 0; i < n; i++){
        sizeH[i] = rooms[i].minH;
        sizeW[i] = rooms[i].minW;

        roomsId.push_back(rooms[i].id);
    }

    const int NSizes = Calculator::NRoomSizes(rooms);
    std::vector<PermLoopRes> perms;
    perms.reserve(NSizes);

    //Main loop
    int count = 0;
    do {
        // std::cout << "#########################" << std::endl << count << std::endl;
        // for(int i = 0; i < n; i++){
        //     std::cout << rooms[i].name << ": " << sizeW[i] << ", " << sizeH[i] << std::endl;
        // }
        // std::cout << "#########################" << std::endl << std::endl << std::endl;

        perms.push_back(roomPerm(sizeH, sizeW, n));
        count++;
        // break;
    } while(Iter::nextRoomSize(rooms, sizeH, sizeW));

    free(sizeH);
    free(sizeW);

    res.roomsId = std::move(roomsId);
    res.perms = std::move(perms);
    return res;
}


/*!
    @brief Iterate over every possible connection between the given rooms 
    @param[in] order, specify the order of the rooms to connect
    @param[in] sizeH Height value of each room setup
    @param[in] sizeW Width value of each room setup
    @param[in] n     number of rooms
    @param[in] NConn Number of possible connections
    @return vector of every successful connection (int)
*/
std::vector<int> Iter::ConnLoop(const std::vector<int>& order, const int *sizeH, const int *sizeW, const int n, const int NConn){
    std::vector<int> result; 
    std::vector<int> ptsX; 
    std::vector<int> ptsY;

    result.reserve(NConn) ;
    ptsX.reserve(n * 2) ;
    ptsY.reserve(n * 2) ;
        
    // std::cout << "perm: ";
    // for(int j = 0; j < n; j++)
    //     std::cout << order[j] << ", ";
    // std::cout << std::endl;

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


        int sum = 0;
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
                if(Iter::check_overlap(ptsX[k*2], ptsX[k*2 + 1], ptsY[k*2], ptsY[k*2 + 1], ptsX[dstIndex], ptsX[dstIndex + 1], ptsY[dstIndex], ptsY[dstIndex + 1])){
                    sucess = false;
                    const int diff = n - j - 1;
                    sum = (1 << (diff * 4)) - 1;
                    break;
                }
            }
        }

        if(sucess){
            result.push_back(i);
            #ifdef OPENCV_ENABLED
            CVHelper::showLayout(ptsX, ptsY, n);
            #endif
        }
        i += sum;
    }

    return result;
}


/*!
    @brief Iterate over every room permutation
    @param[in] sizeH Height value of each room setup
    @param[in] sizeW Width value of each room setup
    @param[in] n     number of rooms
    @return None
*/
PermLoopRes Iter::roomPerm(const int *sizeH, const int *sizeW, const int n){
    std::vector<int> perm;
    for(int i = 0; i < n; i++)
        perm.push_back(i);

    PermLoopRes res;
    const int NPerm = Calculator::Factorial(n);
    const int NConn = Calculator::NConnections(n);

    std::vector<std::vector<int>> perms;
    std::vector<std::vector<int>> conns;
    conns.reserve(NPerm);
    perms.reserve(NPerm);

    // Cycle each permutation
    do {
        perms.push_back(perm);
        conns.push_back(ConnLoop(perm, sizeH, sizeW, n, NConn));
    } while (std::next_permutation(perm.begin(), perm.end()));

    res.conns = std::move(conns);
    res.perms = std::move(perms);
    return res;
}

// // 0--1
// // -  -
// // 2--3