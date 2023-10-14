#include "../lib/viewer.h"
#include "../lib/cvHelper.h"
#include "../lib/storage.h"
#include "../lib/globals.h"
#include <iostream>
#include <math.h>

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
        std::cout << "  -  ";
        for(int j = ptsPerLayout * 2; j < vectorOffset; j++){
                std::cout << arr[i + j] << ", ";
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

// void Viewer::showIndexTable(std::string fullPath){
//     Storage hdd = Storage();
//     std::vector<int> arr = hdd.readVector<int>(fullPath);

//     const int invalid_idx = (int)-1;

//     const int arrSize = arr.size();
//     const int rows = sqrt(arrSize);

//     std::cout << "showIndexTable arrSize: " << arrSize << ", rows: " << rows << std::endl;

//     for(int i = 0; i < rows; i++){
//         std::cout << "row: " << i << std::endl;
//         for(int j = 0; j < rows; j++){
//             const int val = arr[(i * rows) + j];
//             if(val == invalid_idx)
//                 std::cout <<  "_" << ", ";
//             else
//                 std::cout <<  val << ", ";
//         }
//         std::cout << std::endl << std::endl;
//     }
// }

void Viewer::showResults(std::string fullPath){
    Storage hdd = Storage();
    std::vector<int> arr = hdd.readVector<int>(fullPath);

    const size_t arrSize = arr.size();
    const size_t rows = arrSize / __SIZE_RES_DISK;

    std::cout << "showIndexTable arrSize: " << arrSize << ", rows: " << rows << std::endl;

    for(int i = 0; i < arr.size(); i+=__SIZE_RES_DISK){
        // std::cout << "row: " << i << ", max H: " << arr[(i * rows) + __RES_DISK_MAX_H] << ", max W: " << arr[(i * rows) + __RES_DISK_MAX_W] << ", a idx: " << arr[(i * rows) + __RES_DISK_A_IDX] << ", b idx: " << arr[(i * rows) + __RES_DISK_B_IDX] << std::endl;

        for(int j = 0; j < __SIZE_RES_DISK; j++){
            std::cout << arr[i + j] << ", ";
        }
        std::cout << std::endl << std::endl;
    }
}