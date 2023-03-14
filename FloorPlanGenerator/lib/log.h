#ifndef LOG
#define LOG

#include "globals.h"
#include <vector>

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

    /** 
    * @brief console print 1D vector
    * @return None
    */
    template <typename T>
    static void printVector1D(std::vector<T> arr);
};

#endif //LOG