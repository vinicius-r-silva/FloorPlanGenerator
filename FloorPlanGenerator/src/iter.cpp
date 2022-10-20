#include <iostream>
#include <vector>
#include <algorithm>
#include "../lib/iter.h"
#include "../lib/globals.h"
#include "../lib/calculator.h"
#include "../lib/cvHelper.h"
#include <unordered_map>


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
        
        bool validComb = true;
        for (int i = 0; i < k && validComb; ++i){
            bool found = false;
            if(comb[i].depend != 0){
                for (int j = 0; j < k && !found; ++j){
                    if(comb[i].depend == comb[j].id)
                        found = true;
                }
            } else {
                found = true;
            }

            validComb = found;
        }

        if(validComb)
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
bool Iter::nextRoomSize(std::vector<RoomConfig> rooms, int16_t *sizeH, int16_t *sizeW){
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

std::vector<std::vector<int>> Iter::getFilesToCombine(std::vector<int> filesId, std::vector<RoomConfig> rooms){
    std::unordered_map<int, int> umap;
    int fullId = 0;
    for(RoomConfig setup : rooms){
        fullId += setup.id;
    }

    std::vector<std::vector<int>> result;
    result.reserve(filesId.size() / 2);

    for(int i = 0; i < (int)filesId.size(); i++ ){
        int fileId = filesId[i];
        int missingPart = fullId & (~fileId);

        if(umap.contains(fileId) || umap.contains(missingPart))
            continue;

        bool found = false;
        for(int j = 0; j < (int)filesId.size(); j++ ){
            if(filesId[j] == missingPart){
                found = true;
                break;
            }
        }

        if(found){
            umap.insert({fileId, 0});
            umap.insert({missingPart, 0});

            std::vector<int> combination;
            combination.push_back(fileId);
            combination.push_back(missingPart);

            result.push_back(combination);
        }
    }

    return result;
}