#ifndef OPENMP_HELPER
#define OPENMP_HELPER



/** 
 * @brief openMp related functions
*/
class MPHelper
{

public:
    /** 
     * @brief MPHelper Constructor
     * @return None
    */
    MPHelper();
    
    /** 
     * @brief get number of threads in the current system
     * @details https://stackoverflow.com/questions/11071116/i-got-omp-get-num-threads-always-return-1-in-gcc-works-in-icc
     * @return (int) number of threads
    */
    static int omp_thread_count();
};

#endif //OPENMP_HELPER