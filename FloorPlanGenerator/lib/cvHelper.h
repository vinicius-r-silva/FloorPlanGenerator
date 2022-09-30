#ifndef OPENCV_HELPER
#define OPENCV_HELPER

#include <vector>


/** 
 * @brief Handles all opencv functions
*/
class CVHelper
{

public:
    /** 
     * @brief CVHelper Constructor
     * @return None
    */
    CVHelper();
    
    /** 
     * @brief draw and show a layout
     * @param[in] n qtd of rectangles
     * @param[in] ptsX x axis values
     * @param[in] ptsY y axis values
     * @return None
    */
    static void showLayout(const std::vector<int> &ptsX, const std::vector<int> &ptsY, const int n);
};

#endif //OPENCV_HELPER