#include "../lib/combinePostProcess.h"
#include "../lib/combine.h"
#include "../lib/storage.h"
#include "../lib/log.h"
#include <omp.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <chrono>

void CombinePostProcess::postProcess(Storage hdd, const int combId){
    std::chrono::time_point<std::chrono::high_resolution_clock> begin, end;
    begin = std::chrono::high_resolution_clock::now();

    std::vector<int> combFileIds = hdd.getSavedCombinationsPartsCombFileIds(combId);
    // std::cout << "postProcess combId: " << combId << std::endl;
    
    const int nCpuThreads = 3;
    #pragma omp parallel for num_threads(nCpuThreads)
    for(int combFileId : combFileIds){
        std::vector<int> minSizeIds = hdd.getSavedCombinationsPartsMinSizeIds(combId, combFileId);
        // std::cout << "postProcess combFileId: " << combFileId << std::endl;

        // TODO sort result by max size

        for(int minSizeId : minSizeIds){
            std::vector<CombinationResultPart> files = hdd.getSavedCombinationsParts(combId, combFileId, minSizeId);
            std::vector<int> result;

            int maxSizeH = -1;
            int maxSizeW = -1;
            for(CombinationResultPart file : files){
                // std::cout << "kernelLaunchId: " << kernelLaunchId << std::endl;
                std::vector<int> resultPart = hdd.readCombineSplitData(combId, combFileId, minSizeId, file.maxSizeId, file.kernelCount);
                // std::cout << "resultPart: " << resultPart.size() << std::endl;
                result.insert(result.end(), resultPart.cbegin(), resultPart.cend());

                int fileMaxSizeH = file.maxSizeId >> __RES_FILE_LENGHT_BITS;
                int fileMaxSizeW = file.maxSizeId & __RES_FILE_LENGHT_AND_RULE;
                
                if(fileMaxSizeH > maxSizeH)
                    maxSizeH = fileMaxSizeH;

                if(fileMaxSizeW > maxSizeW)
                    maxSizeW = fileMaxSizeW;
            }


            // for(int kernelLaunchId : kernelIds){
            //     // std::cout << "kernelLaunchId: " << kernelLaunchId << std::endl;
            //     std::vector<int> resultPart = hdd.readCombineSplitData(combId, combFileId, minSizeId, kernelLaunchId);
            //     // std::cout << "resultPart: " << resultPart.size() << std::endl;
            //     result.insert(result.end(), resultPart.cbegin(), resultPart.cend());
            // }

            int maxSizeId = (maxSizeH << __RES_FILE_LENGHT_BITS) | maxSizeW;

            long totalSize = result.size();
            // std::cout << combId << ", " << combFileId << ", " << minSizeId << ", " << totalSize  << ", " << totalSize / __SIZE_RES_DISK << ", " << (((double)(totalSize * sizeof(int))) / 1024.0 / 1024.0) << std::endl;  
            hdd.saveCombineResult(result, combId, combFileId, minSizeId, maxSizeId);
        }

        end = std::chrono::high_resolution_clock::now();

        std::chrono::milliseconds duration_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        std::cout << "Elapsed Time milliseconds: " << duration_milliseconds.count() << std::endl;

        std::chrono::minutes duration_minutes = std::chrono::duration_cast<std::chrono::minutes>(end - begin);
        std::cout << "Elapsed Time minutes: " << duration_minutes.count() << std::endl;
    }
}