#include <iostream>
#include <vector>
#include <algorithm>   
#include <omp.h>
#include <string> 
#include <stdlib.h>
#include <cmath>
// #include <boost/numeric/ublas/matrix.hpp>
// #include <boost/numeric/ublas/io.hpp>
// #include <sstream>   
#include "../lib/log.h"
#include "../lib/storage.h"
#include "../lib/iter.h"
#include "../lib/combine.h"
#include "../lib/generate.h"
#include "../lib/calculator.h"
#include "../lib/mpHelper.h"
#include "../cuda/combine.h"

/*!
    @brief create data
    @param[in] n number of rooms per layout
*/
void generateData(const int n) {
    const int NThreads = MPHelper::omp_thread_count();
    std::cout << "NThreads: " << NThreads << std::endl;

    Storage hdd = Storage();
    // std::vector<int> adj = hdd.getAdjValues();
    std::vector<RoomConfig> setups = hdd.getConfigs();

    std::vector<int> allReq = hdd.getReqAdjValues();
    // std::cout << "allReq" << std::endl << allReq << std::endl << std::endl;

    const int reqSize = sqrt(allReq.size());
    
    // counts how many rooms of each class is included in the final layout
    std::vector<int> allReqCount(reqSize, -1);
    for (std::vector<RoomConfig>::iterator it = setups.begin() ; it != setups.end(); ++it){
        RoomConfig room = (RoomConfig)(*it);
        Log::print(room);
        if(allReqCount[room.rPlannyId] == -1)
            allReqCount[room.rPlannyId] = 0;

        allReqCount[room.rPlannyId] += 1;
    }

    Calculator::totalOfCombinations(setups, n);

    std::vector<std::vector<RoomConfig>> allCombs = Iter::getAllComb(setups, n);
    const int NCombs = allCombs.size();
    printf("main NCombs: %d\n\n", NCombs);
    
    const int NReqs = Calculator::lenghtHalfMatrix(n);
    std::vector<int> req(NReqs, 0);

#ifdef MULTI_THREAD
    #pragma omp parallel for num_threads(NThreads) default(none) firstprivate(hdd) shared(allCombs, NCombs)
#endif
    // for(int i = 0; i < NCombs; i++)
    for(int i = 1; i < NCombs; i++)
    {
        const int tid = omp_get_thread_num();
        printf("Thread: %d, i: %d\n", tid, i);

        for(std::size_t k = 0; k < allCombs[i].size(); k++){
           printf("%s, ", allCombs[i][k].name);
        }
        printf("\n");
        for(std::size_t k = 0; k < allCombs[i].size(); k++){
            std::cout << allCombs[i][k].name << ",  ";
        }
        std::cout << std::endl;
        
        // std::cout << "i = " << i << std::endl;

        // hdd.saveResult(Generate::SizeLoop(allCombs[i], adjValues), allCombs[i], n);
        Generate::SizeLoop(reqSize, allReq, allReqCount, allCombs[i]);
        break;
    }
}

static inline std::vector<RoomConfig> getConfigsById(const int layoutId, const std::vector<RoomConfig>& setups){
    std::vector<RoomConfig> result;
    for(int i = 0; i < (int)setups.size(); i++){
        if(setups[i].id & layoutId){
            result.push_back(setups[i]);
        }
    }

    return result;
}

void combineData(){
    Storage hdd = Storage();
    std::vector<RoomConfig> setups = hdd.getConfigs();
    std::vector<int> savedCombs = hdd.getSavedCombinations();
    std::vector<std::vector<int>> filesCombs = Iter::getFilesToCombine(savedCombs, setups);

    for(std::vector<int> fileComb : filesCombs){
        std::cout << fileComb[0] << ", " << fileComb[1] << std::endl;
        
        std::vector<int16_t> layout_a = hdd.readCoreData(fileComb[0]);
        std::vector<int16_t> layout_b = hdd.readCoreData(fileComb[1]);
        
        std::vector<RoomConfig> setupsA = getConfigsById(fileComb[0], setups);
        std::vector<RoomConfig> setupsB = getConfigsById(fileComb[1], setups);

        std::cout << layout_a.size()/(setupsA.size() * 4) << ", " << layout_b.size()/(setupsB.size() * 4) << std::endl << std::endl;
        Combine::getValidLayoutCombs(layout_a, layout_b, setupsA.size(), setupsB.size());
        break;
        Combine::getValidLayoutCombs(layout_b, layout_a, setupsB.size(), setupsA.size());
    }
}

void combineDataGPU(){
    Storage hdd = Storage();
    std::vector<RoomConfig> setups = hdd.getConfigs();
    std::vector<int> savedCombs = hdd.getSavedCombinations();
    std::vector<std::vector<int>> filesCombs = Iter::getFilesToCombine(savedCombs, setups);

    for(std::vector<int> fileComb : filesCombs){
        std::cout << fileComb[0] << ", " << fileComb[1] << std::endl;
        
        std::vector<int16_t> layout_a = hdd.readCoreData(fileComb[0]);
        std::vector<int16_t> layout_b = hdd.readCoreData(fileComb[1]);
        
        std::vector<RoomConfig> setupsA = getConfigsById(fileComb[0], setups);
        std::vector<RoomConfig> setupsB = getConfigsById(fileComb[1], setups);

        std::cout << layout_a.size()/(setupsA.size() * 4) << ", " << layout_b.size()/(setupsB.size() * 4) << std::endl << std::endl;
        gpuHandler::createPts(layout_a, layout_b);
        break;
    }
}

/*!
    @brief Main Function
    @return if there are no erros returns 0 
*/
int main(){
    generateData(3);
    // combineData();
    // combineDataGPU();
    // std::vector<int> a;
    // Cuda_Combine::launchGPU(a, a, 0, 0);
    return 0;
}