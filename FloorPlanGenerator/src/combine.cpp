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

// static inline void combineCore(int *connA, int *connB, int *sizeHA, int *sizeWA, int *sizeHB, int *sizeWB, const std::vector<int>& permA, const std::vector<int>& permB){

// }

void Combine::getValidLayoutCombs(const std::vector<int>& a, const std::vector<int>& b, const int n_a, const int n_b){

    const int n_final = n_a + n_b;
    const int vectorOffset_a = n_a * 4;
    const int vectorOffset_b = n_b * 4;
    // std::cout << "n_a: " << n_a << ", n_b: " << n_b << std::endl;

    std::vector<int> ptsX(n_final * 2, 0); 
    std::vector<int> ptsY(n_final * 2, 0);

    const int ptsPerLayout_a = n_a * 2;
    const int ptsPerLayout_b = n_b * 2;

    for(int i = vectorOffset_a; i <= (int)a.size(); i += vectorOffset_a){
    
        // if(i % 480 == 0)
        //     std::cout << "i: " << i << std::endl;
        // std::cout << "a: ";
        // for(int j = i; j < i + n_a*4; j++)   
        //     std::cout << a[j] << ", ";
        // std::cout << std::endl;

        for(int j = 0; j < ptsPerLayout_a; j++){
            ptsX[j] = a[i - vectorOffset_a + (j * 2)];
            ptsY[j] = a[i - vectorOffset_a + (j * 2) + 1];
        }
        
        for(int j = 0; j < (int)b.size(); j += vectorOffset_b){
    
            // std::cout << "b: ";
            // for(int k = j; k < j + n_b*4; k++)   
            //     std::cout << b[k] << ", ";
            // std::cout << std::endl;

            for(int k = 0; k < 16; k++){                
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
                    srcX = a[i - 4];
                else 
                    srcX = a[i - 2];
                    
                if(srcConn == 0 || srcConn == 1)
                    srcY = a[i - 3];
                else 
                    srcY = a[i - 1];

                const int diffX = srcX - dstX;
                const int diffY = srcY - dstY;

                for(int l = 0; l < ptsPerLayout_b; l++){
                    ptsX[l + ptsPerLayout_a] = b[j + l * 2] + diffX;
                    ptsY[l + ptsPerLayout_a] = b[j + l * 2 + 1] + diffY;
                }

                // std::cout << "i: " << i << ", j: " << j << ", k: " << k << std::endl;
                // for(int l = 0; l < n_final; l++){
                //     std::cout << "\t(" << ptsX[l * 2] << ", " << ptsY[l * 2] << "), (" << ptsX[l * 2 + 1] << ", " << ptsY[l * 2 + 1] << ")" << std::endl;
                // }
                CVHelper::showLayout(ptsX, ptsY);
            }            
            // break;
        }
        // break;
    }
}