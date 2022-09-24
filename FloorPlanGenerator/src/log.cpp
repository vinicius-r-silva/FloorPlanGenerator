#include <iostream>
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
    std::cout << s.id << " " << s.name << ": H (" << s.minH << 
                 " - " << s.maxH << "), : W (" << s.minW << 
                 " - " << s.maxW << "), E: " << s.numExtensions << 
                 " step: " << s.step << std::endl;
}