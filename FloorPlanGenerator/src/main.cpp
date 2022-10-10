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
#include "../lib/generate.h"
#include "../lib/calculator.h"
#include "../lib/mpHelper.h"


void generateData() {
    const int n = 3;
    const int NThreads = MPHelper::omp_thread_count();
    std::cout << "NThreads: " << NThreads << std::endl;

    Storage hdd = Storage();
    std::vector<RoomConfig> setups = hdd.getConfigs();
    setups.pop_back();
    setups.pop_back();
    setups.pop_back();

    for (std::vector<RoomConfig>::iterator it = setups.begin() ; it != setups.end(); ++it)
        Log::print((RoomConfig)(*it));


    Calculator::totalOfCombinations(setups, n);
    // return;

    std::vector<std::vector<RoomConfig>> allCombs = Iter::getAllComb(setups, n);
    const int NCombs = allCombs.size();
    printf("main NCombs: %d\n", NCombs);

#ifdef MULTI_THREAD
    #pragma omp parallel for num_threads(NThreads) default(none) firstprivate(hdd) shared(allCombs, NCombs)
#endif
    for(int i = 0; i < NCombs; i++)
    {
        const int tid = omp_get_thread_num();
        printf("Thread: %d, i: %d\n", tid, i);
        // for(std::size_t k = 0; k < allCombs[i].size(); k++){
        //    printf("%s, ", allCombs[i][k].name);
        // }
        // printf("\n");
        // for(std::size_t k = 0; k < allCombs[i].size(); k++){
        //     std::cout << allCombs[i][k].name << ",  ";
        // }
        // std::cout << std::endl;
        
        // std::cout << "i = " << i << std::endl;

        hdd.saveResult(Generate::SizeLoop(allCombs[i]), allCombs[i], n);
        // Generate::SizeLoop(allCombs[i]);
        

        // break;
    }
}

void combineData(){
    const int n = 6;
    Storage hdd = Storage();
    std::vector<int> savedCombs = hdd.getSavedCombinations();
    

}

/*!
    @brief Main Function
    @return if there are no erros returns 0 
*/
int main(){
    generateData();
    // combineData();
    return 0;
}