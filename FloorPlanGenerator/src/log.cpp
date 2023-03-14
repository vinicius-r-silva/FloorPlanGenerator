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
                 " - " << s.maxW << "),\tE: " << s.numExtensions << 
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

template void Log::printVector1D<int>(std::vector<int>);