#include "../lib/combine.h"
#include "../lib/cvHelper.h"
#include "../lib/globals.h"
#include <iostream>

/** 
 * @brief Storage Constructor
 * @return None
*/
Combine::Combine(){
}


void Combine::getValidLayoutCombs(const std::vector<int16_t>& a, const std::vector<int16_t>& b, const int n_a, const int n_b){

    const int n_final = n_a + n_b;
    const int vectorOffset_a = n_a * 4 + 1;
    const int vectorOffset_b = n_b * 4 + 1;
    // std::cout << "n_a: " << n_a << ", n_b: " << n_b << std::endl;

    std::vector<int16_t> ptsX(n_final * 2, 0); 
    std::vector<int16_t> ptsY(n_final * 2, 0);

    const int ptsPerLayout_a = n_a * 2;
    const int ptsPerLayout_b = n_b * 2;

    std::vector<int> iArray;
    std::vector<int> jArray;
    std::vector<int> kArray;

    for(int i = vectorOffset_a; i <= (int)a.size(); i += vectorOffset_a){
        std::cout << "i: " << i << std::endl;
    
        for(int j = 0; j < ptsPerLayout_a; j++){
            ptsX[j] = a[i - vectorOffset_a + (j * 2)];
            ptsY[j] = a[i - vectorOffset_a + (j * 2) + 1];
        }

        // for(int k = 1; k < 2; k++){       
        //     for(int j = 0; j < 1; j += vectorOffset_b){   
   
        for(int j = 0; j < (int)b.size(); j += vectorOffset_b){   
            for(int k = 0; k < 16; k++){         
                // std::cout << "j: " << j << std::endl;
                const int srcConn = k & 0b11;
                const int dstConn = (k >> 2) & 0b11;

                if(srcConn == dstConn)
                    continue;

                int dstX = 0;
                int dstY = 0;
                if(dstConn == 0 || dstConn == 2)
                    dstX = b[j];
                else 
                    dstX = b[j + 2];
                    
                if(dstConn == 0 || dstConn == 1)
                    dstY = b[j + 1];
                else 
                    dstY = b[j + 3];

                int srcX = 0;
                int srcY = 0;
                if(srcConn == 0 || srcConn == 2)
                    srcX = a[i - 5];
                else 
                    srcX = a[i - 3];
                    
                if(srcConn == 0 || srcConn == 1)
                    srcY = a[i - 4];
                else 
                    srcY = a[i - 2];

                const int diffX = srcX - dstX;
                const int diffY = srcY - dstY;
                // const int diffX = 0;
                // const int diffY = 0;

                for(int l = 0; l < ptsPerLayout_b; l++){
                    ptsX[l + ptsPerLayout_a] = b[j + l * 2] + diffX;
                    ptsY[l + ptsPerLayout_a] = b[j + l * 2 + 1] + diffY;
                }

                std::cout << std::endl << std::endl << std::endl << iArray.size() << std::endl;
                std::cout << "i: " << i << ", j: " << j << ", k: " << k << std::endl;
                std::cout << "diffX: " << diffX << ", diffY: " << diffY << std::endl;
                int minX = 99999; int maxX = -99999;
                int minY = 99999; int maxY = -99999;
                for(int l = 0; l < n_final; l++){
                    if(ptsX[l * 2] < minX)
                        minX = ptsX[l * 2];
                    if(ptsX[l * 2 + 1] < minX)
                        minX = ptsX[l * 2 + 1];

                    if(ptsX[l * 2] > maxX)
                        maxX = ptsX[l * 2];
                    if(ptsX[l * 2 + 1] > maxX)
                        maxX = ptsX[l * 2 + 1];

                    if(ptsY[l * 2] < minY)
                        minY = ptsY[l * 2];
                    if(ptsY[l * 2 + 1] < minY)
                        minY = ptsY[l * 2 + 1];

                    if(ptsY[l * 2] > maxY)
                        maxY = ptsY[l * 2];
                    if(ptsY[l * 2 + 1] > maxY)
                        maxY = ptsY[l * 2 + 1];

                    std::cout << "\t(" << ptsX[l * 2] << ", " << ptsY[l * 2] << "), (" << ptsX[l * 2 + 1] << ", " << ptsY[l * 2 + 1] << ")" << std::endl;
                }
                std::cout << "H: " << maxY - minY << ", W: " << maxX - minX << std::endl;
                std::cout << "Adj A: " << a[i - 1] << ", Adj B: " << b[j + vectorOffset_b - 1] << std::endl;

                // ptsX.erase(ptsX.begin() + 6, ptsX.end());
                // ptsY.erase(ptsY.begin() + 6, ptsY.end());
                
                int dir = CVHelper::showLayoutMove(ptsX, ptsY);
                if(dir == -1 && iArray.size() == 0){
                    // i -= vectorOffset_a;
                    // j -= vectorOffset_b;
                    k -= 1;
                }
                else if(dir == -1 && iArray.size() > 0){
                    i = iArray.back(); iArray.pop_back(); 
                    // i = iArray.back() - vectorOffset_a; iArray.pop_back(); 

                    j = jArray.back(); jArray.pop_back(); 
                    // j = jArray.back() - vectorOffset_b; jArray.pop_back(); 

                    // k = kArray.back(); kArray.pop_back(); 
                    k = kArray.back() - 1; kArray.pop_back(); 
                } else {
                    iArray.push_back(i);
                    jArray.push_back(j);
                    kArray.push_back(k);
                }
                // break;
            }            
            // break;
        }
        // break;
    }
}