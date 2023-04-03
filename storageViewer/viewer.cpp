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

std::vector<int> getSavedCombinations() {
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

void showTwoLayouts(){
    
}

int main(){
    updateProjectDir();
    // std::vector<int> filesIds = getSavedCombinations();
    int idA = 11;
    int idB = 11;

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
        for(int ptIdx = 0; ptIdx < lPts; ptIdx++){
            ptsXA.push_back(ptsA[i * lSize + ptIdx*2]);
            ptsYA.push_back(ptsA[i * lSize + ptIdx*2 + 1]);
            std::cout << "(" << ptsXA[ptIdx] << ", " << ptsYA[ptIdx] << "), ";
        }
        std::cout << std::endl;

        for(int j = 0; j < qtdB; j++){
            std::cout << "\t";
            ptsXB.clear();
            for(int ptIdx = 0; ptIdx < lPts; ptIdx++){
                ptsXB.push_back(ptsB[j * lSize + ptIdx*2]);
                ptsYB.push_back(ptsB[j * lSize + ptIdx*2 + 1]);
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
    return 0;
}