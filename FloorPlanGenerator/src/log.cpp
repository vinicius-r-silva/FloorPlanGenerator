#include <iostream>
#include <vector>
#include "../lib/log.h"
#include "../lib/globals.h"

/** 
 * @brief Storage Constructor
 * @return None
*/
Log::Log(){
}
    
/** 
 * @brief console print RoomConfig object
 * @return None
*/
void Log::print(RoomConfig s){
    std::cout << s.id << " " << s.name << ":\tH (" << s.minH << 
                 " - " << s.maxH << "),\tW (" << s.minW << 
                 " - " << s.maxW << "),\tArea (" << s.minArea << 
                 " - " << s.maxArea << ")\tE: " << s.numExtensions << 
                 "\tstep: " << s.step <<
                 "\tdepend: " << s.depend << 
                 "\tPlannyId: " << s.rPlannyId << std::endl;
}
    
/** 
 * @brief console print 1D vector
 * @return None
*/
template <typename T>
void Log::printVector1D(std::vector<T> arr){
    for(T val : arr){
        std::cout << val << ", ";
    }
   std::cout <<  std::endl;
}
    
/** 
 * @brief console print 2D vector
 * @return None
*/
void Log::printVector2D(std::vector<std::vector<RoomConfig>> arr){
    for(std::vector<RoomConfig> innerArray : arr){
        for(RoomConfig val : innerArray){
            Log::print(val);
        }
        std::cout <<  std::endl;
    }
   std::cout <<  std::endl;
}

template void Log::printVector1D<int>(std::vector<int>);
template void Log::printVector1D<int16_t>(std::vector<int16_t>);
template void Log::printVector1D<std::string>(std::vector<std::string>);