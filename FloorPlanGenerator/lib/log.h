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

    /** 
    * @brief console print 2D vector
    * @return None
    */
    static void printVector2D(std::vector<std::vector<RoomConfig>> arr);
};

#endif //LOG