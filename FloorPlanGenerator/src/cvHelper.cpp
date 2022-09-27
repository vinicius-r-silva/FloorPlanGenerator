#include <iostream>
#include <opencv2/opencv.hpp>

#include "../lib/cvHelper.h"
#include "../lib/globals.h"

/** 
 * @brief Storage Constructor
 * @return None
*/
CVHelper::CVHelper(){
}
    
/** 
 * @brief draw and show a layout
 * @param[in] n qtd of rectangles
 * @param[in] ptsX x axis values
 * @param[in] ptsY y axis values
 * @return None
*/
void CVHelper::showLayout(const std::vector<short> &ptsX, const std::vector<short> &ptsY, const int n){
#ifdef OPENCV_ENABLED
    cv::Mat fundo = cv::Mat::zeros(cv::Size(500, 500), CV_8UC3);
    for(int i = 0; i < n; i++){
        cv::Scalar color = cv::Scalar(35 + (220 * ((i + 1) & 0b1)), 35 + (220 * ((i + 1) & 0b10)), 35 + (220 * ((i + 1) & 0b100)));   
        cv::rectangle(fundo, cv::Point(ptsX[i*4]*5 + 200, ptsY[i*4]*5 + 200), cv::Point(ptsX[i*4 + 3]*5 + 200, ptsY[i*4 + 3]*5 + 200), color, 2, 8, 0);
    }

    cv::namedWindow("tela", cv::WINDOW_AUTOSIZE );
    cv::imshow("tela", fundo);
    cv::waitKey(1);
    // while(cv::waitKey(30) != 27);
#endif
}