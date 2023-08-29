#include "../lib/viewer.h"
#include "../lib/cvHelper.h"
#include "../lib/storage.h"
#include "../lib/globals.h"
#include <iostream>

/** 
 * @brief Storage Constructor
 * @return None
*/
Viewer::Viewer(){
}

void Viewer::showCoreResults(const std::vector<int16_t>& arr, const int n){
    const int vectorOffset = n * 4 + 1;
    const int ptsPerLayout = n * 2;
    std::vector<int16_t> ptsX(n * 2, 0); 
    std::vector<int16_t> ptsY(n * 2, 0);
    std::cout << "vectorOffset: " << vectorOffset << ", ptsPerLayout: " << ptsPerLayout << std::endl;

    for(int i = 0; i <= (int)arr.size(); i += vectorOffset){    
        std::cout << "i: " << i << ", idx: " << i / vectorOffset << std::endl;
        for(int j = 0; j < ptsPerLayout; j++){
            ptsX[j] = arr[i + (j * 2)];
            ptsY[j] = arr[i + (j * 2) + 1];

            std::cout << "(" << ptsX[j] << ", " << ptsY[j] << "), ";
        }
        std::cout << std::endl;

        int dir = CVHelper::showLayoutMove(ptsX, ptsY);

        if(dir == -1 && i == 0){
            i -= vectorOffset;
        } else if(dir == -1){
            i -= (vectorOffset*2);
        }
    }
}



void Viewer::showFileResults(std::string fullPath, int arrayOffset, int layoutSize){
    Storage hdd = Storage();
    std::vector<int16_t> arr = hdd.readVector<int16_t>(fullPath);

    int ptsPerLayout = layoutSize / 2;
    std::vector<int16_t> ptsX(ptsPerLayout, 0); 
    std::vector<int16_t> ptsY(ptsPerLayout, 0);
    std::cout << "arrayOffset: " << arrayOffset << ", layoutSize: " << layoutSize << ", ptsPerLayout: " << ptsPerLayout << std::endl;

    for(int i = 0; i <= (int)arr.size(); i += arrayOffset){    
        std::cout << "i: " << i << ", idx: " << i / arrayOffset << std::endl;
        for(int j = 0; j < ptsPerLayout; j++){
            ptsX[j] = arr[i + (j * 2)];
            ptsY[j] = arr[i + (j * 2) + 1];

            std::cout << "(" << ptsX[j] << ", " << ptsY[j] << "), ";
        }
        std::cout << std::endl;
        

        int dir = CVHelper::showLayoutMove(ptsX, ptsY);

        if(dir == -1 && i == 0){
            i -= arrayOffset;
        } else if(dir == -1){
            i -= (arrayOffset*2);
        }
    }
}