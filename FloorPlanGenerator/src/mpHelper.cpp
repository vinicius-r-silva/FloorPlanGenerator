#include "../lib/mpHelper.h"
#include "../lib/globals.h"
#include <omp.h>


/** 
 * @brief Storage Constructor
 * @return None
*/
MPHelper::MPHelper(){
}
    
/** 
 * @brief get number of threads in the current system
 * @details https://stackoverflow.com/questions/11071116/i-got-omp-get-num-threads-always-return-1-in-gcc-works-in-icc
 * @return (int) number of threads
*/
int MPHelper::omp_thread_count() {
    #ifdef OPENCV_ENABLED
        return 1;
    #endif
    
    #ifndef MULTI_THREAD
        return 1;
    #endif

    int n = 0;
    #pragma omp parallel reduction(+:n)
        n += 1;
    
    return n;
}