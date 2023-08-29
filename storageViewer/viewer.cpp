#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

std::string _projectDir;

void updateProjectDir(){
    std::string res = ""; //Result string
    const std::filesystem::path currPath = std::filesystem::current_path();
    
    //Iterate over every folder of the current path until it reachs the FloorPlanGenerator folder
    for (auto it = currPath.begin(); it != currPath.end(); ++it){
        res += (*it);
        res += "/";

        if (res.find("FloorPlanGenerator") != std::string::npos) 
            break;
    }

    //Cleanup result
    if (res.rfind("//", 0) == 0)
        res.erase(0,1);

    if(res.length() > 0)
        res.pop_back();

    _projectDir = res;
}

std::vector<int> getSavedCoreCombinations() {
    std::vector<int> result;
    std::string path = _projectDir + "/FloorPlanGenerator/storage";

    for (const auto & entry : std::filesystem::directory_iterator(path)){
        std::string fileName = entry.path().stem();
        std::string extension = entry.path().extension();

        if(extension.compare(".dat") == 0){
            result.push_back(stoi(fileName));
        }
        std::cout << fileName << "  " << extension <<std::endl;
    }

    return result;
}

std::vector<int16_t> readCoreData(int id){
    const std::string filename = _projectDir + "/FloorPlanGenerator/storage/" + std::to_string(id) + ".dat";
    
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);

    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg() / sizeof(int16_t);
    file.seekg(0, std::ios::beg);

    // read the data:
    std::vector<int16_t> fileData(fileSize);
    file.read((char*) &fileData[0], fileSize * sizeof(int16_t));

    return fileData;
}


void showLayoutMove(const std::vector<int16_t> &ptsX, const std::vector<int16_t> &ptsY, std::string windowName){
    int scale = 5;
    const int screenH = 500;
    const int screenW = 500;
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

    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE );
    cv::imshow(windowName, fundo);
    cv::waitKey(1);
    // while(cv::waitKey(30) != 27);
}

void showTwoLayouts(const int idA, const int idB){
    std::vector<int16_t> ptsA = readCoreData(idA);
    std::vector<int16_t> ptsB = readCoreData(idB);

    const int n = 3;
    const int lPts = n * 2;
    const int lSize = lPts * 2;
    const int qtdA = ptsA.size() / lSize;
    const int qtdB = ptsB.size() / lSize;
    // const int qtdB = 4;

    std::vector<int16_t> ptsXA; ptsXA.reserve(lPts);
    std::vector<int16_t> ptsYA; ptsYA.reserve(lPts);
    std::vector<int16_t> ptsXB; ptsXB.reserve(lPts);
    std::vector<int16_t> ptsYB; ptsYB.reserve(lPts);

    for(int i = 0; i < qtdA; i++){
        ptsXA.clear();
        ptsYA.clear();
        for(int ptIdx = 0; ptIdx < lPts; ptIdx++){
            ptsXA.push_back(ptsA[i * lSize + ptIdx*2]);
            ptsYA.push_back(ptsA[i * lSize + ptIdx*2 + 1]);
            std::cout << "(" << ptsXA[ptIdx] << ", " << ptsYA[ptIdx] << "), ";
        }
        std::cout << std::endl;

        for(int j = 0; j < qtdB; j++){
            std::cout << "\t";
            ptsXB.clear();
            ptsYB.clear();
            for(int ptIdx = 0; ptIdx < lPts; ptIdx++){
                ptsXB.push_back(ptsB[(j * lSize) + (ptIdx*2)]);
                ptsYB.push_back(ptsB[(j * lSize) + (ptIdx*2) + 1]);
                std::cout << "(" << ptsXB[ptIdx] << ", " << ptsYB[ptIdx] << "), ";
            }
            std::cout << std::endl;

            showLayoutMove(ptsXA, ptsYA, "A " + std::to_string(idA));
            showLayoutMove(ptsXB, ptsYB, "B " + std::to_string(idB));

            while(1){
                int c = cv::waitKey(0);
                if(c == 100){
                    break;
                }

                if(c == 97 || c == 27){
                    j -= 2;
                    if(j == -2){
                        j = -1;
                        i -= 2;
                        if(i == -2)
                            i = -1;
                    }
                    break;
                }
            }
        }
    }    
}

void showComb(const int idA, const int idB){
    std::vector<int16_t> ptsA = readCoreData(idA);
    std::vector<int16_t> ptsB = readCoreData(idB);

    const int n = 3;
    const int lPts = n * 2;
    const int lSize = lPts * 2;
    const int qtdA = ptsA.size() / lSize;
    const int qtdB = ptsB.size() / lSize;
    // const int qtdB = 4;

    for(int i = 0; i < 30; i++){
        std::cout << ptsB[i] << ", ";
        if(i % lSize == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;

    std::vector<int16_t> ptsXA; ptsXA.reserve(lPts);
    std::vector<int16_t> ptsYA; ptsYA.reserve(lPts);
    std::vector<int16_t> ptsXB; ptsXB.reserve(lPts);
    std::vector<int16_t> ptsYB; ptsYB.reserve(lPts);
    std::vector<int16_t> resX; resX.reserve(lPts * 2);
    std::vector<int16_t> resY; resY.reserve(lPts * 2);

    for(int i = 0; i < qtdA; i++){
        ptsXA.clear();
        ptsYA.clear();
        for(int ptIdx = 0; ptIdx < lPts; ptIdx++){
            ptsXA.push_back(ptsA[i * lSize + ptIdx*2]);
            ptsYA.push_back(ptsA[i * lSize + ptIdx*2 + 1]);
            // std::cout << "(" << ptsXA[ptIdx] << ", " << ptsYA[ptIdx] << "), ";
        }
        // std::cout << std::endl;

        for(int k = 1; k < 16; k++){
        // for(int j = 0; j < 12; j++){
        for(int j = 0; j < qtdB; j++){
            ptsXB.clear();
            ptsYB.clear();
            for(int ptIdx = 0; ptIdx < lPts; ptIdx++){
                ptsXB.push_back(ptsB[(j * lSize) + (ptIdx*2)]);
                ptsYB.push_back(ptsB[(j * lSize) + (ptIdx*2) + 1]);
                // std::cout << "(" << ptsXB[ptIdx] << ", " << ptsYB[ptIdx] << "), ";
            }
            // std::cout << std::endl;

                std::cout << std::endl;
                std::cout << "k: " << k << ", j: " << j << ", i: " << i << std::endl;
                int srcConn = k % 4;
                int dstConn = k / 4;

                if(srcConn == dstConn)
                    continue;

                int diffX = 0;
                int diffY = 0;

                if(srcConn == 0 || srcConn == 2)
                    diffX = ptsXA[lPts - 2];
                else
                    diffX = ptsXA[lPts - 1];
                
                if(srcConn == 0 || srcConn == 1)
                    diffY = ptsYA[lPts - 2];
                else
                    diffY = ptsYA[lPts - 1];

                if(dstConn == 0 || dstConn == 2)
                    diffX -= ptsXB[0];
                else
                    diffX -= ptsXB[1];
                
                if(dstConn == 0 || dstConn == 1)
                    diffY -= ptsYB[0];
                else
                    diffY -= ptsYB[1];
                    
                resX.clear();
                resY.clear();
                for(int a = 0; a < lPts; a++){
                    resX.push_back(ptsXA[a]);
                    resY.push_back(ptsYA[a]);
                    std::cout << "(" << resX[resX.size() - 1] << ", " << resY[resY.size() - 1] << "), ";
                }
                std::cout << "| ";
                for(int a = 0; a < lPts; a++){
                    resX.push_back(ptsXB[a] + diffX);
                    resY.push_back(ptsYB[a] + diffY);
                    std::cout << "(" << resX[resX.size() - 1] << ", " << resY[resY.size() - 1] << "), ";
                }
                // std::cout << std::endl;
                // for(int a = 0; a < lPts; a++){
                //     std::cout << "(" << ptsXB[a] << ", " << ptsYB[a] << "), ";
                // }
                std::cout << std::endl;
                std::cout << std::endl;

                showLayoutMove(resX, resY, std::to_string(idA) + " - " + std::to_string(idB));

                while(1){
                    int c = cv::waitKey(0);
                    if(c == 100){
                        break;
                    }

                    // if(c == 97 || c == 27){
                    //     k -= 2;
                    //     if((k + 1) / 4 == (k + 1) % 4) 
                    //         k -= 1;

                    //     if(k == -2){
                    //         k = 16;
                    //         j -= 2;
                    //         if(j == -2){
                    //             j = qtdB;
                    //             i -= 2;
                    //             if(i == -2)
                    //                 i = -1;
                    //         }
                    //     }
                    //     break;
                    // }

                    if(c == 97 || c == 27){
                        j -= 2;
                        if(j <= -2){
                            j = -1;

                            k -= 2;
                            if((k + 1) / 4 == (k + 1) % 4) 
                                k -= 1;

                            if(k <= -1){
                                k = 0;
                                i -= 2;
                                if(i <= -1)
                                    i = 0;
                            }
                        }
                        break;
                    }
                }
            }

            // showLayoutMove(ptsXA, ptsYA, "A " + std::to_string(idA));
            // showLayoutMove(ptsXB, ptsYB, "B " + std::to_string(idB));

        }
    }    
}

int main(){
    updateProjectDir();

    int idA = 52;
    int idB = 11;
    // showTwoLayouts(idA, idB);
    showComb(idA, idB);

    return 0;
}