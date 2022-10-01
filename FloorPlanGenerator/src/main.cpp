#include <iostream>
#include <vector>
#include <algorithm>   
#include <omp.h>
#include <string> 
#include <stdlib.h>
// #include <sstream>   

#ifdef OPENCV_ENABLED 
    #include <opencv2/opencv.hpp>
#endif

#include "../lib/log.h"
#include "../lib/storage.h"
#include "../lib/iter.h"



/*!
    @brief Main Function
    @return if there are no erros returns 0 
*/
int main(){
    Storage hdd = Storage();
    std::vector<RoomConfig> setups = hdd.getConfigs();

    std::vector<std::vector<RoomConfig>> allCombs = Iter::getAllComb(setups, 3);
    for(std::size_t i = 0; i < allCombs.size(); i++){
        // for(std::size_t k = 0; k < allCombs[i].size(); k++){
        //     std::cout << allCombs[i][k].name << ",  ";
        // }
        // std::cout << std::endl;
        
        Iter::SizeLoop(allCombs[i]);
        // break;
    }
    return 0;
}