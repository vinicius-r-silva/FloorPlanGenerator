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
#include "../lib/search.h"
#include "../cuda/combine.h"
#include "../cuda/generate.h"
#include "../lib/viewer.h"

/*!
    @brief counts how many rooms of each class is included in the final layout
    @param[in] setups rooms information
    @return vector count how many of each class (class 0 -> allReqCount[0], ...)
*/
std::vector<int> countReqClasses(std::vector<RoomConfig> setups, int reqSize){
    std::vector<int> allReqCount(reqSize, -1);
    for (std::vector<RoomConfig>::iterator it = setups.begin() ; it != setups.end(); ++it){
        RoomConfig room = (RoomConfig)(*it);
        // Log::print(room);
        if(allReqCount[room.rPlannyId] == -1)
            allReqCount[room.rPlannyId] = 0;

        allReqCount[room.rPlannyId] += 1;
    }

    return allReqCount;
}

// /*!
//     @brief create data
//     @param[in] n number of rooms per layout
// */
// void generateData(const int n) {
//     const int NThreads = MPHelper::omp_thread_count();
//     std::cout << "NThreads: " << NThreads << std::endl;

//     Storage hdd = Storage();
//     std::vector<RoomConfig> setups = hdd.getConfigs();

//     std::vector<int> allReq = hdd.getReqAdjValues();
//     // std::cout << "allReq" << std::endl << allReq << std::endl << std::endl;

//     const int reqSize = sqrt(allReq.size());
    
//     // counts how many rooms of each class is included in the final layout
//     std::vector<int> allReqCount = countReqClasses(setups, reqSize);

//     Calculator::totalOfCombinations(setups, n);

//     std::vector<std::vector<RoomConfig>> allCombs = Iter::getAllComb(setups, n);
//     const int NCombs = allCombs.size();
//     printf("main NCombs: %d\n\n", NCombs);
//     Log::printVector2D(allCombs);

// #ifdef MULTI_THREAD
//     #pragma omp parallel for num_threads(NThreads) default(none) firstprivate(hdd) shared(allCombs, NCombs, allReqCount, allReq, reqSize, n)
// #endif
//     // for(int i = 0; i < NCombs; i++)
//     for(int i = 1; i < NCombs; i++)
//     {
//         const int tid = omp_get_thread_num();
//         printf("Thread: %d, i: %d\n", tid, i);

//         // for(std::size_t k = 0; k < allCombs[i].size(); k++){
//         //    printf("%s, ", allCombs[i][k].name);
//         // }
//         // printf("\n");
//         // for(std::size_t k = 0; k < allCombs[i].size(); k++){
//         //     std::cout << allCombs[i][k].name << ",  ";
//         // }
//         // std::cout << std::endl;
        
//         // std::cout << "i = " << i << std::endl;
        
//         hdd.saveResult(Generate::SizeLoop(reqSize, allReq, allReqCount, allCombs[i]), allCombs[i], n);
//         // Generate::SizeLoop(reqSize, allReq, allReqCount, allCombs[i]);

//         // break;
//     }
// }


void generateDataGpu() {
    Storage hdd = Storage();
    std::vector<RoomConfig> setups = hdd.getConfigs();
    std::vector<int> allReq = hdd.getReqAdjValues();
    
    const int reqSize = sqrt(allReq.size());
    std::vector<int> allReqCount = countReqClasses(setups, reqSize);
	std::cout << "allReqCount main: ";
    for(int val : allReqCount){
        std::cout << val << ", ";
    }
   	std::cout <<  std::endl;

    std::vector<std::vector<RoomConfig>> allCombs = Iter::getAllComb(setups, __GENERATE_N);
    const int NCombs = allCombs.size();

    for(int i = 1; i < NCombs; i++)
    {
        CudaGenerate::generateCuda(allCombs[i], allReq, allReqCount, reqSize);
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
    std::vector<int> savedCombs = hdd.getSavedCoreCombinations();
    std::vector<std::vector<int>> filesCombs = Iter::getFilesToCombine(savedCombs, setups);

    for(std::vector<int> fileComb : filesCombs){
        std::cout << fileComb[0] << ", " << fileComb[1] << std::endl;
        
        std::vector<int16_t> layout_a = hdd.readCoreData(fileComb[0]);
        std::vector<int16_t> layout_b = hdd.readCoreData(fileComb[1]);
        
        std::vector<RoomConfig> setupsA = getConfigsById(fileComb[0], setups);
        std::vector<RoomConfig> setupsB = getConfigsById(fileComb[1], setups);

        std::cout << layout_a.size()/(setupsA.size() * 4) << ", " << layout_b.size()/(setupsB.size() * 4) << std::endl << std::endl;
        Combine::getValidLayoutCombs(layout_a, layout_b, setupsA.size(), setupsB.size());
        // break;
        Combine::getValidLayoutCombs(layout_b, layout_a, setupsB.size(), setupsA.size());
    }
}

void showCoreResults(){
    Storage hdd = Storage();
    std::vector<RoomConfig> setups = hdd.getConfigs();
    std::vector<int> savedCombs = hdd.getSavedCoreCombinations();

    for(int combId : savedCombs){
        std::cout << "combId: " << combId << std::endl;
        
        std::vector<int16_t> layout = hdd.readCoreData(combId);
        std::vector<RoomConfig> setup = getConfigsById(combId, setups);

        Viewer::showCoreResults(layout, setup.size());
    }
}

void combineDataGPU(){
    Storage hdd = Storage();
    std::vector<RoomConfig> setups = hdd.getConfigs();
    std::vector<int> savedCombs = hdd.getSavedCoreCombinations();

    //TODO check if all of req are present during the gpu combination
    std::vector<int> allReq = hdd.getReqAdjValues();
    // Log::printVector1D(allReq);

    std::vector<std::vector<int>> filesCombs = Iter::getFilesToCombine(savedCombs, setups);

    for(std::vector<int> fileComb : filesCombs){        
        std::cout << "fileComb[0]: " << fileComb[0] << ", fileComb[1]: " << fileComb[1] << std::endl;
        std::vector<int16_t> layout_a = hdd.readCoreData(fileComb[0]);
        std::vector<int16_t> layout_b = hdd.readCoreData(fileComb[1]);
        
        // std::vector<RoomConfig> setupsA = getConfigsById(fileComb[0], setups);
        // std::vector<RoomConfig> setupsB = getConfigsById(fileComb[1], setups);

        std::string resultPath = hdd.getResultPath();

        // TODO: rotate layout b with layout a, or change cuda code to process the two ways os combination
        gpuHandler::createPts(layout_a, layout_b, allReq, resultPath, fileComb[0], fileComb[1]);
        break;
    }
}

void showReults(){
    Storage hdd = Storage();
    std::vector<RoomConfig> setups = hdd.getConfigs();
    std::vector<int> savedResults = hdd.getSavedResults();

    std::string imagesPath = hdd.getImagesPath();

    for(int resultName : savedResults){
        const int b_comb = resultName >> 16;
        const int a_comb = resultName ^ (b_comb << 16);

        std::vector<int16_t> layout_a = hdd.readCoreData(a_comb);
        std::vector<int16_t> layout_b = hdd.readCoreData(b_comb);
        std::vector<int> cudaResult = hdd.readResultData(resultName);

        std::vector<RoomConfig> setupsA = getConfigsById(a_comb, setups);
        std::vector<RoomConfig> setupsB = getConfigsById(b_comb, setups);
        
        Search::ShowContent(cudaResult, layout_a, layout_b, setupsA.size(), setupsB.size(), imagesPath);
        break;
    }
}

/*!
    @brief Main Function
    @return if there are no erros returns 0 
*/
int main(){
    // Process::processResult((int*)0, 0);

    // generateData(3);
    // combineData();
    // combineDataGPU();
    // showReults();
    // showCoreResults();
    generateDataGpu();
    // Viewer::showFileResults("/home/ribeiro/Documents/FloorPlanGenerator/FloorPlanGenerator/storage/temp/generate.dat", __GENERATE_RES_LENGHT, __GENERATE_RES_LAYOUT_LENGHT);

    
    return 0;
}