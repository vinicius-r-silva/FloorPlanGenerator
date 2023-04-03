#include <iostream>
// #ifdef OPENCV_ENABLED 
#include <opencv2/opencv.hpp>
// #endif

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
 * @param[in] ptsX x axis values
 * @param[in] ptsY y axis values
 * @return None
*/
void CVHelper::showLayout(const std::vector<int16_t> &ptsX, const std::vector<int16_t> &ptsY){
#ifdef OPENCV_ENABLED
    int scale = 5;
    const int screenH = 600;
    const int screenW = 600;
    const int n = (int)ptsX.size()/2;

    int minX = 999999, maxX = 0;
    int minY = 999999, maxY = 0;
    for(int i = 0; i < (int)ptsX.size(); i++){
        int x = ptsX[i];
        int y = ptsY[i];
        if(x < minX)
            minX = x;
        if(x > maxX)
            maxX = x;

        if(y < minY)
            minY = y;
        if(y > maxY)
            maxY = y;
    }

    if((screenW / (maxX - minX)) < ((scale*8) / 10))
        scale = (8 * (screenW / (maxX - minX))) / 10;

    if((screenH / (maxY - minY)) < ((scale*8) / 10))
        scale = (8 * (screenH / (maxY - minY))) / 10;

    int offsetX = (screenW/2 - ((maxX + minX) * scale)/2);
    int offsetY = (screenH/2 - ((maxY + minY) * scale)/2);    

    cv::Mat fundo = cv::Mat::zeros(cv::Size(screenW, screenH), CV_8UC3);
    for(int i = 0; i < n; i++){
        cv::Scalar color = cv::Scalar(55 + (200 * ((i + 1) & 0b1)), 
                                      55 + (200 * ((i + 1) & 0b10)), 
                                      55 + (200 * ((i + 1) & 0b100)));   
        cv::rectangle(fundo, cv::Point(ptsX[i*2]*scale + offsetX, ptsY[i*2]*scale + offsetY), 
                      cv::Point(ptsX[i*2 + 1]*scale + offsetX, ptsY[i*2 + 1]*scale + offsetY), 
                      color, 2, 8, 0);
    }

    cv::namedWindow("tela", cv::WINDOW_AUTOSIZE );
    cv::imshow("tela", fundo);
    cv::waitKey(1);
    while(cv::waitKey(30) != 27);
#else
    std::cout << ptsX[0] << ptsY[0] << std::endl;
#endif
}


int CVHelper::showLayoutMove(const std::vector<int16_t> &ptsX, const std::vector<int16_t> &ptsY){
#ifdef OPENCV_ENABLED
    int scale = 5;
    const int screenH = 1000;
    const int screenW = 1000;
    const int n = (int)ptsX.size()/2;

    int minX = 999999, maxX = 0;
    int minY = 999999, maxY = 0;
    for(int i = 0; i < (int)ptsX.size(); i++){
        int x = ptsX[i];
        int y = ptsY[i];
        if(x < minX)
            minX = x;
        if(x > maxX)
            maxX = x;

        if(y < minY)
            minY = y;
        if(y > maxY)
            maxY = y;
    }

    if((screenW / (maxX - minX)) < ((scale*8) / 10))
        scale = (8 * (screenW / (maxX - minX))) / 10;

    if((screenH / (maxY - minY)) < ((scale*8) / 10))
        scale = (8 * (screenH / (maxY - minY))) / 10;

    int offsetX = (screenW/2 - ((maxX + minX) * scale)/2);
    int offsetY = (screenH/2 - ((maxY + minY) * scale)/2);    

    cv::Mat fundo = cv::Mat::zeros(cv::Size(screenW, screenH), CV_8UC3);
    for(int i = 0; i < n; i++){
        cv::Scalar color = cv::Scalar(55 + (200 * ((i + 1) & 0b1)), 
                                      55 + (200 * ((i + 1) & 0b10)), 
                                      55 + (200 * ((i + 1) & 0b100)));   
        cv::rectangle(fundo, cv::Point(ptsX[i*2]*scale + offsetX, ptsY[i*2]*scale + offsetY), 
                      cv::Point(ptsX[i*2 + 1]*scale + offsetX, ptsY[i*2 + 1]*scale + offsetY), 
                      color, 2, 8, 0);
    }

    cv::namedWindow("tela", cv::WINDOW_AUTOSIZE );
    cv::imshow("tela", fundo);
    cv::waitKey(1);

    while(1){
        int c = cv::waitKey(0);
        if(c == 100 || c == 27){
           return 1;
        }

        if(c == 97){
           return -1;
        }
    }
    // while(cv::waitKey(30) != 27);
#else
    std::cout << ptsX[0] << ptsY[0] << std::endl;
    return 1;
#endif
}


int CVHelper::showLayoutMove(const std::vector<int16_t> &layouts, const int n){
#ifdef OPENCV_ENABLED
#else
    std::cout << layouts[0] << n << std::endl;
    return 1;
#endif
}