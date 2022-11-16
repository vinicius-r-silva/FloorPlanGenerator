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
    std::cout << s.id << " " << s.name << ":\tH (" << s.minH << 
                 " - " << s.maxH << "),\tW (" << s.minW << 
                 " - " << s.maxW << "),\tE: " << s.numExtensions << 
                 "\tstep: " << s.step <<
                 "\tdepend: " << s.depend << 
                 "\tPlannyId: " << s.rPlannyId << std::endl;
}