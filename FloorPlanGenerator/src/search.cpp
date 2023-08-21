#include "../lib/search.h"
#include "../lib/cvHelper.h"
#include "../lib/globals.h"
#include <iostream>
#include <string>

/** 
 * @brief Search Constructor
 * @return None
*/
Search::Search(){
}

bool Search::check_adjacency(const int a_up, const int a_down, const int a_left, const int a_right, const int b_up, const int b_down, const int b_left, const int b_right){
 return (((a_down == b_up || a_up == b_down) && ((a_right > b_left && a_right <= b_right) || 
            (a_left < b_right && a_left >= b_left) || (a_left <= b_left && a_right >= b_right))) ||  
         ((a_left == b_right || a_right == b_left) && ((a_down > b_up && a_down <= b_down) || 
            (a_up < b_down && a_up >= b_up) || (a_up <= b_up && a_down >= b_down))));
}

bool Search::check_overlap(const int a_up, const int a_down, const int a_left, const int a_right, const int b_up, const int b_down, const int b_left, const int b_right){
	if(((a_down > b_up && a_down <= b_down) ||
	(a_up  >= b_up && a_up < b_down)) &&
	((a_right > b_left && a_right <= b_right) ||
	(a_left  >= b_left && a_left  <  b_right) ||
	(a_left  <= b_left && a_right >= b_right))){
		return true;
	}

	else if(((b_down > a_up && b_down <= a_down) ||
	(b_up >= a_up && b_up < a_down)) &&
	((b_right > a_left && b_right <= a_right) ||
	(b_left  >= a_left && b_left  <  a_right) ||
	(b_left  <= a_left && b_right >= a_right))){
		return true;
	}

	else if(((a_right > b_left && a_right <= b_right) ||
	(a_left >= b_left && a_left < b_right)) &&
	((a_down > b_up && a_down <= b_down) ||
	(a_up  >= b_up && a_up   <  b_down) ||
	(a_up  <= b_up && a_down >= b_down))){
		return true;
	}

	else if(((b_right > a_left && b_right <= a_right) ||
	(b_left >= a_left && b_left < a_right)) &&
	((b_down > a_up && b_down <= a_down) ||
	(b_up  >= a_up && b_up   <  a_down) ||
	(b_up  <= a_up && b_down >= a_down))){
		return true;
	}

	return false;
}

bool Search::CalculatePts(std::vector<int16_t>& ptsX, std::vector<int16_t>& ptsY, const std::vector<int16_t>& a, const std::vector<int16_t>& b, int a_offset, int b_offset, const int n_a, const int n_b, const int conn, const int diffH, const int diffW){
    const int layout_size_a = (n_a * 4) + 1;
    // const int layout_size_b = n_a * 4 + 1;
    const int ptsPerLayout_a = n_a * 2;
    const int ptsPerLayout_b = n_b * 2;

    // a_offset *= layout_size_a;
    // b_offset *= layout_size_b;
    const int a_offset_end = a_offset + layout_size_a;
    
    const int srcConn = conn & 0b11;
    const int dstConn = (conn >> 2) & 0b11;

    // std::cout << "a_offset: " << a_offset << ", b_offset: " << b_offset << ", n_a: " << n_a << ", n_b: " << n_b << ", conn: " << conn << std::endl;
    // std::cout << "srcConn: " << srcConn << ", dstConn: " << dstConn << std::endl;

    if(srcConn == dstConn){
        return false;
    }

	int minH = 5000, maxH = -5000;
	int minW = 5000, maxW = -5000;

    for(int i = 0; i < ptsPerLayout_a; i++){
        ptsX[i] = a[a_offset + (i * 2)];
        ptsY[i] = a[a_offset + (i * 2) + 1];

        if(ptsX[i] > maxW)
            maxW = ptsX[i];
        if(ptsX[i] < minW)
            minW = ptsX[i];

        if(ptsY[i] > maxH)
            maxH = ptsY[i];
        if(ptsY[i] < minH)
            minH = ptsY[i];
    }

    // std::cout << "a: ";
    // for(int i = 0; i < n_a * 4; i++){
    //     std::cout << a[a_offset + i] << ", ";
    // }


    // std::cout << std::endl << "b: ";
    // for(int i = 0; i < n_b * 4; i++){
    //     std::cout << b[b_offset + i] << ", ";
    // }
    // std::cout << std::endl;
    // std::cout << "offsetX: " << offsetX << ", offsetY: " << offsetY << std::endl;

    int dstX = 0;
    int dstY = 0;
    if(dstConn == 0 || dstConn == 2)
        dstX = b[b_offset];
    else 
        dstX = b[b_offset + 2];
        
    if(dstConn == 0 || dstConn == 1)
        dstY = b[b_offset + 1];
    else 
        dstY = b[b_offset + 3];

    int srcX = 0;
    int srcY = 0;
    if(srcConn == 0 || srcConn == 2)
        srcX = a[a_offset_end - 5];
    else    
        srcX = a[a_offset_end - 3];
        
    if(srcConn == 0 || srcConn == 1)
        srcY = a[a_offset_end - 4];
    else 
        srcY = a[a_offset_end - 2];

    const int offsetX = srcX - dstX;
    const int offsetY = srcY - dstY;
    // std::cout << "srcX: " << srcX << ", dstX: " << dstX << ", srcY: " << srcY << ", dstY: " << dstY << std::endl;
    // std::cout << "offsetX: " << offsetX << ", offsetY: " << offsetY << std::endl;
    // std::cout << std::endl;

    for(int i = 0; i < ptsPerLayout_b; i++){
        const int idx = i + ptsPerLayout_a;
        ptsX[idx] = b[b_offset + (i * 2)] + offsetX;
        ptsY[idx] = b[b_offset + (i * 2) + 1] + offsetY;

        if(ptsX[idx] > maxW)
            maxW = ptsX[idx];
        if(ptsX[idx] < minW)
            minW = ptsX[idx];
            
        if(ptsY[idx] > maxH)
            maxH = ptsY[idx];
        if(ptsY[idx] < minH)
            minH = ptsY[idx];
    }

    // std::cout << "pts: ";
    // for(int i = 0; i < ptsPerLayout_a + ptsPerLayout_b; i++){
    //     std::cout << "(" << ptsX[i] << ", " << ptsY[i] << "), ";
    // }
    // std::cout << std::endl;
    // std::cout << "maxW: " << maxW << ", minW: " << minW << ", maxH: " << maxH << ", minH: " << minH << std::endl;
    // std::cout << "maxW - minW: " << maxW - minW << ", maxH - minH: " << maxH - minH << std::endl;

    if(maxW - minW != diffW || maxH - minH != diffH)
        return false;


    for(int i = 0; i < n_a + n_b; i++){
		const int a_left = ptsX[i * 2];
		const int a_right = ptsX[(i * 2) + 1];

		const int a_up = ptsY[i * 2];
		const int a_down = ptsY[(i * 2) + 1];

        for(int j = i + 1; j < n_a + n_b; j++){
            const int b_left = ptsX[j * 2];
            const int b_right = ptsX[(j * 2) + 1];
            
            const int b_up = ptsY[j * 2];
            const int b_down = ptsY[(j * 2) + 1];

            // std::cout << "a_left: " << a_left << ", a_right: " << a_right << ", a_up: " << a_up << ", a_down: " << a_down << std::endl;
            // std::cout << "b_left: " << b_left << ", b_right: " << b_right << ", b_up: " << b_up << ", b_down: " << b_down << std::endl;
            // std::cout << "check_overlap: " << Search::check_overlap(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right) << std::endl;
            // std::cout << "check_adjacency: " << Search::check_adjacency(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right) << std::endl;
            // std::cout << std::endl;

            if(Search::check_overlap(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right))
				return false;
			
			// if(Search::check_adjacency(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right))
            //     return false;
        }
    }
    // std::cout << std::endl;
    // std::cout << "-----------------------------------";
    // std::cout << std::endl;
    // std::cout << std::endl;

    return true;
}

void Search::ShowContent(const std::vector<int>& cudaResult, const std::vector<int16_t>& a, const std::vector<int16_t>& b, const int n_a, const int n_b, std::string imagesPath){
    std::vector<int16_t> ptsX((n_a + n_b) * 2, 0);
    std::vector<int16_t> ptsY((n_a + n_b) * 2, 0);

    std::vector<int> last_i;
    std::vector<int> last_j;

    for(unsigned long i = 0; i < cudaResult.size(); i+= __SIZE_RES){
		int diffH = cudaResult[i];
		int diffW = cudaResult[i + 1];
		int a_layout_idx = cudaResult[i + 2];
		int b_layout_idx = cudaResult[i + 3];


        // if(diffH != 40 || diffW != 95)
        //     continue;

        if(i < 472)
            continue;

        for(int j = 0; j < __N_CONN; j++){
            const int conn = j + 1 + j/4;
            std::cout << "1 i: " << i << ", j: " << j << ", conn: " << conn << std::endl;

            // std::fill (ptsX.begin(), ptsX.end(), 0);
            // std::fill (ptsY.begin(), ptsY.end(), 0);
            if(Search::CalculatePts(ptsX, ptsY, a, b, a_layout_idx, b_layout_idx, n_a, n_b, conn, diffH, diffW)){
                // std::cout << std::endl;
                std::cout << "i: " << i << ", j: " << j << ", conn: " << conn << std::endl;
                std::cout << "diffH: " << diffH << ", diffW: " << diffW << ", a_layout_idx: " << a_layout_idx << ", b_layout_idx: " << b_layout_idx << std::endl;

                int dir = CVHelper::showLayoutMove(ptsX, ptsY);
                if(dir == -1 && last_i.size() == 0){
                    j = -1;
                }
                else if(dir == -1){
                    i = last_i.back(); last_i.pop_back(); 
                    j = last_j.back() - 1; last_j.pop_back(); 

                    diffH = cudaResult[i];
                    diffW = cudaResult[i + 1];
                    a_layout_idx = cudaResult[i + 2];
                    b_layout_idx = cudaResult[i + 3];
                } else {
                    last_i.push_back(i);
                    last_j.push_back(j);
                }

                // std::cout << "3 i: " << i << ", j: " << j << ", conn: " << conn << std::endl;
                std::cout << std::endl;
                // std::cout << std::endl;

                // std::string fullPath = imagesPath + "/" + std::to_string(diffH) + "_" + std::to_string(diffW) + "_" + std::to_string(a_layout_idx) + "_" + std::to_string(b_layout_idx) + "_" + std::to_string(j) + ".png";
                // std::cout << fullPath << std::endl;
                // CVHelper::saveImage(ptsX, ptsY, fullPath);
            }
            // std::cout << std::endl;
            
            // CVHelper::showLayout(ptsX, ptsY);
            // break;
        }
    }
}