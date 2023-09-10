#ifndef OPENCV_HELPER
#define OPENCV_HELPER

#include <vector>
#include <string>
#include "globals.h"

#ifdef OPENCV_ENABLED 
#include <opencv2/opencv.hpp>
#endif


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
     * @param[in] ptsX x axis values
     * @param[in] ptsY y axis values
     * @return None
    */
    static void showLayout(const std::vector<int16_t> &ptsX, const std::vector<int16_t> &ptsY);

    static int showLayoutMove(const std::vector<int16_t> &ptsX, const std::vector<int16_t> &ptsY);

#ifdef OPENCV_ENABLED
    static void createLayoutImage(cv::Mat& fundo, const std::vector<int16_t> &ptsX, const std::vector<int16_t> &ptsY);
#endif

    static void saveImage(const std::vector<int16_t> &ptsX, const std::vector<int16_t> &ptsY, std::string fullPath);

};

#endif //OPENCV_HELPER