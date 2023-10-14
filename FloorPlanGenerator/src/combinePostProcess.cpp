#include "../lib/combinePostProcess.h"
#include "../lib/combine.h"
#include "../lib/storage.h"
#include "../lib/log.h"
#include <iostream>
#include <vector>
#include <math.h>

void CombinePostProcess::postProcess(Storage hdd, const int combId){
    std::vector<int> minSizeIds = hdd.getSavedCombinationsPartsMinSizeIds(combId);
    long totalSize = 0;

    for(int minSizeId : minSizeIds){
        std::vector<int> kernelIds = hdd.getSavedCombinationsPartsKernelIds(combId, minSizeId);

        std::vector<int> result;
        for(int kernelLaunchId : kernelIds){
            std::vector<int> resultPart = hdd.readCombineSplitData(combId, minSizeId, kernelLaunchId);
            result.insert(result.end(), resultPart.cbegin(), resultPart.cend());
            // std::cout << "reading combId: " << combId << ", minSizeId: " << minSizeId << ", kernelLaunchId: " << kernelLaunchId << ", layouts: " << resultPart.size() / __SIZE_RES_DISK << ", size: " << ((resultPart.size() * sizeof(int)) / 1024 / 1024) << " MB" << std::endl;  

        }

        totalSize += result.size();
        // std::cout << "post process combId: " << combId << ", minSizeId: " << minSizeId << ", layouts: " << result.size() / __SIZE_RES_DISK << ", size: " << (((double)(result.size() * sizeof(int))) / 1024.0 / 1024.0) << " MB" << std::endl;  

        hdd.saveCombineResult(result, combId, minSizeId);
    }

    std::cout << "post process combId: " << combId  << ", layouts: " << totalSize / __SIZE_RES_DISK << ", size: " << (((double)(totalSize * sizeof(int))) / 1024.0 / 1024.0) << " MB" << std::endl;  

    // std::cout << "total Layouts: " << totalSize / __SIZE_RES_DISK << ", size: " << (((double)(totalSize * sizeof(int))) / 1024.0 / 1024.0) << " MB" << std::endl << std::endl << std::endl;  
}