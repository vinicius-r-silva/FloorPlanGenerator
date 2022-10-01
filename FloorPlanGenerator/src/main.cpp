#include <iostream>
#include <vector>
#include <algorithm>   
#include <omp.h>
#include <string> 
#include <stdlib.h>
// #include <sstream>   
// #include "../lib/log.h"
#include "../lib/storage.h"
#include "../lib/iter.h"
#include "../lib/calculator.h"


void totalOfCombinations();

/*!
    @brief Main Function
    @return if there are no erros returns 0 
*/
int main(){
    const int n = 3;
    Storage hdd = Storage();
    std::vector<RoomConfig> setups = hdd.getConfigs();
    // Calculator::totalOfCombinations(setups, n);

    std::vector<std::vector<RoomConfig>> allCombs = Iter::getAllComb(setups, n);
    
    std::vector<SizeLoopRes> sizes;
    sizes.reserve(allCombs.size());

    for(std::size_t i = 0; i < allCombs.size(); i++){
        // for(std::size_t k = 0; k < allCombs[i].size(); k++){
        //     std::cout << allCombs[i][k].name << ",  ";
        // }
        // std::cout << std::endl;
        
        sizes.push_back(Iter::SizeLoop(allCombs[i]));
        break;
    }
    return 0;
}