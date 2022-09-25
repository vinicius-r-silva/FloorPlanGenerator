#ifndef LOG
#define LOG

#include "log.h"
#include "globals.h"

/** 
 * @brief Handles all console's read/write
*/
class Log
{

public:
    /** 
     * @brief Storage Constructor
     * @return None
    */
    Log();
    
    /** 
     * @brief console print RoomConfig object
     * @return None
    */
    static void print(RoomConfig setup);
};

#endif //LOG