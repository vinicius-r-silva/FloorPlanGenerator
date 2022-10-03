#include <iostream>
#include <vector>
#include <algorithm>   
#include <omp.h>
#include <string> 
#include <stdlib.h>
// #include <sstream>   
#include "../lib/log.h"
#include "../lib/storage.h"
#include "../lib/iter.h"
#include "../lib/calculator.h"


// https://stackoverflow.com/questions/11071116/i-got-omp-get-num-threads-always-return-1-in-gcc-works-in-icc
int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
        n += 1;
    
    return n;
}

/*!
    @brief Main Function
    @return if there are no erros returns 0 
*/
int main(){
    const int n = 3;
    #ifdef OPENCV_ENABLED
    const int NThreads = 1;
    #endif

    #ifndef OPENCV_ENABLED
    const int NThreads = omp_thread_count();
    #endif

    Storage hdd = Storage();
    std::vector<RoomConfig> setups = hdd.getConfigs();
    
    for (std::vector<RoomConfig>::iterator it = setups.begin() ; it != setups.end(); ++it)
        Log::print((RoomConfig)(*it));

    Calculator::totalOfCombinations(setups, n);

    std::vector<std::vector<RoomConfig>> allCombs = Iter::getAllComb(setups, n);

    const int NCombs = allCombs.size();
    SizeLoopRes *sizes = (SizeLoopRes*)calloc(NCombs, sizeof(SizeLoopRes));

    #pragma omp parallel for num_threads(NThreads) default(none) shared(allCombs, sizes, NCombs)
    for(int i = 0; i < NCombs; i++)
    {
        // const int tid = omp_get_thread_num();
        // printf("Thread: %d, i: %d\n", tid, i);
        // for(std::size_t k = 0; k < allCombs[i].size(); k++){
        //     std::cout << allCombs[i][k].name << ",  ";
        // }
        // std::cout << std::endl;
        
        // std::cout << "i = " << i << std::endl;

        sizes[i] = Iter::SizeLoop(allCombs[i]);
    }
    return 0;
}