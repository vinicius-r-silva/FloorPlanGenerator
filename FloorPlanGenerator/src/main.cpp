#include <iostream>
#include <vector>
#include <algorithm>   
#include <omp.h>
#include <string> 
#include <stdlib.h>
#include <cmath>
#include <chrono>
#include <iomanip>
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
#include "../cuda/combineHandler.h"
#include "../cuda/generateHandler.h"
#include "../lib/viewer.h"
#include "../lib/combinePostProcess.h"

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
    std::cout << "generateDataGpu init" << std::endl;
    Storage hdd = Storage();
    std::vector<RoomConfig> setups = hdd.getConfigs();
    std::vector<int> allReq = hdd.getReqAdjValues();

    hdd.deleteSavedCoreResults();
    
    const int reqSize = sqrt(allReq.size());
    std::vector<int> allReqCount = countReqClasses(setups, reqSize);

    std::vector<std::vector<RoomConfig>> allCombs = Iter::getAllComb(setups, __GENERATE_N);
    const int NCombs = allCombs.size();

    GenerateHandler handler = GenerateHandler();

    for(int i = 0; i < NCombs; i++)
    {            
        int combId  = 0;
        for(int j = 0; j < __GENERATE_N; j++){
            combId += allCombs[i][j].id;
        }

	    std::cout << std::endl << std::endl << std::endl;
        std::cout << "combId: " << combId << std::endl;

        for(RoomConfig config : allCombs[i]){
            Log::print(config);
        }

        // if(combId != 14)
        //     continue;

        handler.generate(allCombs[i], allReqCount, allReq, reqSize, combId, hdd);
        // break;
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
    std::vector<int> savedCombs = hdd.getSavedCores();
    std::vector<std::vector<int>> filesCombs = Iter::getFilesToCombine(savedCombs, setups);

    // for(std::vector<int> fileComb : filesCombs){
    //     std::cout << fileComb[0] << ", " << fileComb[1] << std::endl;
        
    //     std::vector<int16_t> layout_a = hdd.readCoreData(fileComb[0]);
    //     std::vector<int16_t> layout_b = hdd.readCoreData(fileComb[1]);
        
    //     std::vector<RoomConfig> setupsA = getConfigsById(fileComb[0], setups);
    //     std::vector<RoomConfig> setupsB = getConfigsById(fileComb[1], setups);

    //     std::cout << layout_a.size()/(setupsA.size() * 4) << ", " << layout_b.size()/(setupsB.size() * 4) << std::endl << std::endl;
    //     Combine::getValidLayoutCombs(layout_a, layout_b, setupsA.size(), setupsB.size());
    //     // break;
    //     Combine::getValidLayoutCombs(layout_b, layout_a, setupsB.size(), setupsA.size());
    // }
}

void showCoreResults(){
    Storage hdd = Storage();
    std::vector<RoomConfig> setups = hdd.getConfigs();
    std::vector<int> savedCombs = hdd.getSavedCores();
    std::sort (savedCombs.begin(), savedCombs.end()); // 12 32 45 71(26 33 53 80)

    for(int combId : savedCombs){
        std::vector<int> fileIds = hdd.getSavedCoreFiles(combId);

        for(int fileId : fileIds){
            std::cout << std::endl << std::endl << std::endl;
            std::cout << "combId: " << combId << ", fileId: " << fileId << std::endl;

            std::vector<RoomConfig> setup = getConfigsById(combId, setups);
            for(RoomConfig room : setup){
                Log::print(room);
            }

            std::vector<int16_t> layout = hdd.readCoreData(combId, fileId);
            Viewer::showLayouts(layout, setup.size(), 1);
        }
    }
}

void combineDataGPU(){
    std::chrono::time_point<std::chrono::high_resolution_clock> begin, end;
    begin = std::chrono::high_resolution_clock::now();
    std::cout << std::fixed << std::setprecision(3);

    Storage hdd = Storage();
    std::vector<RoomConfig> setups = hdd.getConfigs();
    std::vector<int> savedCombs = hdd.getSavedCores();
    CombineHandler handler = CombineHandler();

    // hdd.deleteSavedCombinedResultsParts();

    //TODO check if all of req are present during the gpu combination
    std::cout << "fcombineDataGPU getReqAdjValues" << std::endl;
    std::vector<int> allReq = hdd.getReqAdjValues();
    // Log::printVector1D(allReq);

    std::cout << "fcombineDataGPU getFilesToCombine" << std::endl;
    std::vector<std::vector<int>> filesCombs = Iter::getFilesToCombine(savedCombs, setups);

    for(std::vector<int> fileComb : filesCombs){     
        std::cout << "fileComb[0]: " << fileComb[0] << ", fileComb[1]: " << fileComb[1] << std::endl;

        std::vector<int> layout_a_files_ids = hdd.getSavedCoreFiles(fileComb[0]);
        std::vector<int> layout_b_files_ids = hdd.getSavedCoreFiles(fileComb[1]);

        for(int layout_a_file_id : layout_a_files_ids){
            for(int layout_b_file_id : layout_b_files_ids){
                std::cout << "layout_a_file_id: " << layout_a_file_id << ", layout_b_file_id: " << layout_b_file_id << std::endl;
                
                std::vector<RoomConfig> config_a = hdd.getConfigsById(fileComb[0]);
                std::vector<RoomConfig> config_b = hdd.getConfigsById(fileComb[1]);

                bool invertLayouts = false;
                if(config_a.size() != __COMBINE_N_A && config_a.size() != __COMBINE_N_B){
                    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
                    std::cout << "!!!!!!!config size error!!!!!!!" << std::endl;
                    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
                    return;
                }

                if(config_a.size() != __COMBINE_N_A){
                    invertLayouts = true;
                }

                std::vector<int16_t> layout_a = hdd.readCoreData(fileComb[0], layout_a_file_id);
                std::vector<int16_t> layout_b = hdd.readCoreData(fileComb[1], layout_b_file_id);

                std::cout << "a: " << std::endl;
                for(RoomConfig room : config_a){
                    Log::print(room);
                }
                
                std::cout << std::endl << std::endl << "b: " << std::endl;
                for(RoomConfig room : config_b){
                    Log::print(room);
                }

                std::cout << "invertLayouts: " << invertLayouts << std::endl;
                std::cout << std::endl << std::endl;

                // const int filesdId = (layout_a_file_id << __COMBINE_NAME_ROOMS_ID_SHIFT) | layout_b_file_id;

                // // std::string resultPath = hdd.getResultPath();

                if(invertLayouts){
                    const int filesdId = (layout_b_file_id << __COMBINE_NAME_ROOMS_ID_SHIFT) | layout_a_file_id;
                    handler.combine(config_b, config_a, layout_b, layout_a, filesdId, allReq, hdd);
                }
                else {
                    const int filesdId = (layout_a_file_id << __COMBINE_NAME_ROOMS_ID_SHIFT) | layout_b_file_id;
                    handler.combine(config_a, config_b, layout_a, layout_b, filesdId, allReq, hdd);
                }
                // // return;
                // break;
            }
            // break;
        }
        // break;
    }

    // hdd.updateCombinationList();

    end = std::chrono::high_resolution_clock::now();

    std::chrono::milliseconds duration_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Elapsed Time milliseconds: " << duration_milliseconds.count() << std::endl;

    std::chrono::minutes duration_minutes = std::chrono::duration_cast<std::chrono::minutes>(end - begin);
    std::cout << "Elapsed Time minutes: " << duration_minutes.count() << std::endl;
}

// void showReults(){
    // Storage hdd = Storage();
    // std::vector<RoomConfig> setups = hdd.getConfigs();
    // std::vector<int> savedResults = hdd.getSavedResults();

    // std::string imagesPath = hdd.getImagesPath();

    // for(int resultName : savedResults){
    //     const int b_comb = resultName >> 16;
    //     const int a_comb = resultName ^ (b_comb << 16);

    //     std::vector<int16_t> layout_a = hdd.readCoreData(a_comb);
    //     std::vector<int16_t> layout_b = hdd.readCoreData(b_comb);
    //     std::vector<int> cudaResult = hdd.readResultData(resultName);

    //     std::vector<RoomConfig> setupsA = getConfigsById(a_comb, setups);
    //     std::vector<RoomConfig> setupsB = getConfigsById(b_comb, setups);
        
    //     Search::ShowContent(cudaResult, layout_a, layout_b, setupsA.size(), setupsB.size(), imagesPath);
    //     break;
    // }
// }

// void test(std::vector<int16_t> result){
// 	const size_t resultSize = result.size();
//     Log::printVector1D<int16_t>(result);
//     std::cout << "resultSize: " << resultSize << std::endl;

// 	size_t dst = 0;
// 	for(; dst < resultSize && result[dst] != -1; dst += 3);

// 	size_t i = dst;
// 	size_t src_init = 0, src_end = 0;

// 	while(true){
//         std::cout << "dst: " << dst;
//         // for(; result[i] == -1 && i < resultSize; i += 3){};
// 	    for(; i < resultSize && result[i] == -1; i += 3);

// 		src_init = i;
//         std::cout << ", src_init: " << src_init;

//         // for(; result[i] != -1 && i < resultSize; i += 3){};
// 	    for(; i < resultSize && result[i] != -1; i += 3);


// 		src_end = i;
//         std::cout << ", src_end: " << src_end << std::endl;
//         // std::cout << ", src_end: " << src_end << std::endl;
// 		if(src_init == src_end)
// 			break;

// 		std::copy(result.begin() + src_init, result.begin() + src_end, result.begin() + dst);
// 		dst += src_end - src_init;

// 		if(i >= resultSize)
// 			break;
// 	}
// 	result.resize(dst);

//     std::cout << "dst: " << dst << ", result.size(): " << result.size() << std::endl;
//     Log::printVector1D<int16_t>(result);
//     std::cout << std::endl << std::endl << std::endl;
// }

void postProcess(){
    std::cout << "main postProcess init" << std::endl;
    
    Storage hdd = Storage();
    std::vector<int> savedPartsCombIds = hdd.getSavedCombinationsPartsCombIds();

    hdd.deleteSavedCombinedResults();

    for(int combId : savedPartsCombIds){
        CombinePostProcess::postProcess(hdd, combId);
        // break;
    }

    hdd.updateCombinationList();
}

void search(){
    Storage hdd = Storage();
    hdd.deleteSavedImages();

    std::vector<int16_t> inputShape(4, 0);

    inputShape[__UP] = 0;
    inputShape[__LEFT] = 0;
    inputShape[__DOWN] = 40;
    inputShape[__RIGHT] = 95;

    Search::moveToCenterOfMass(inputShape);

    Search::getLayouts(inputShape, hdd);
    // Search::getLayouts(hdd, 95, 40);
}

/*!
    @brief Main Function
    @return if there are no erros returns 0 
*/
int main(){
    // Process::processResult((int*)0, 0);

    // generateData(3);
    // generateDataGpu();
    // combineData();
    combineDataGPU();
    postProcess();
    //
    // search();


    // std::cout << "showResults. sizeIdx " << 163935 << ", min H: " << (163935 >> __RES_FILE_LENGHT_BITS) << ", min W: " << (163935 & __RES_FILE_LENGHT_AND_RULE) << std::endl;
    // Viewer::showResults("/home/ribeiro/Documents/FloorPlanGenerator/FloorPlanGenerator/storage/combined/parts/3407883_163935_0.dat");
    // Viewer::showIndexTable("/home/ribeiro/Documents/FloorPlanGenerator/FloorPlanGenerator/storage/combined/parts/3407883_0_table.dat");
    // showReults();
    // showCoreResults();
    // Viewer::showFileResults("/home/ribeiro/Documents/FloorPlanGenerator/FloorPlanGenerator/storage/core/21_0.dat", __GENERATE_RES_LENGHT, __GENERATE_RES_LAYOUT_LENGHT);
    // 
    // Storage hdd = Storage();
    
    return 0;
}