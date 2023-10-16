#include "../lib/search.h"
#include "../lib/cvHelper.h"
#include "../lib/globals.h"
#include <iostream>
#include <string>
#include "../cuda/combineHandler.h"
#include "../lib/viewer.h"
#include "../lib/log.h"

// TODO somehow limit search by room sizes as well

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


// std::map<int, std::vector<int>> Search::getValidCombIdx(Storage hdd, const int h, const int w, const int tolerance){
//     std::map<int, std::vector<int>> result;

//     std::vector<int> combIds = hdd.getSavedCombinationsCombIds();
//     for(int combId : combIds){
//         result[combId] = getValidCombIdxFromComb(hdd, combId, h, w, tolerance);

//         std::cout << "read " << combId << ", layouts: " << result[combId].size() / 2 << ", ram MB: " << result[combId].size() / 1024 / 1024 << std::endl;  
//     }

//     return result;
// }

std::vector<int> Search::getValidCombIdxFromComb(Storage hdd, const int combId, const int combFileId, const int h, const int w, const int tolerance){
    std::vector<int> result;

    const int start_h = h - tolerance;
    const int start_w = w - tolerance;
    const int end_h = h + tolerance;
    const int end_w = w + tolerance;

    std::vector<CombinationResult> resultFiles = hdd.getSavedCombinations(combId, combFileId);

    // std::cout << std::endl << std::endl << std::endl << "getValidCombIdxFromComb: " << std::endl;
    std::cout << "combId: " << combId << ", combFileId: " << combFileId << std::endl;

    for(CombinationResult file : resultFiles){
        int file_min_h = file.minSizeId >> __RES_FILE_LENGHT_BITS;
        int file_min_w = file.minSizeId & __RES_FILE_LENGHT_AND_RULE;

        int file_max_h = file.maxSizeId >> __RES_FILE_LENGHT_BITS;
        int file_max_w = file.maxSizeId & __RES_FILE_LENGHT_AND_RULE;


        // if(combId == 917553)
        //     std::cout << std::endl << "file_min_h: " << file_min_h << ", file_min_w: " << file_min_w << ", file_max_h: " << file_max_h << ", file_max_w: " << file_max_w << std::endl;

        if(file_min_h > end_h || file_min_w > end_w || file_max_h < start_h || file_max_w < start_w)
            continue;

        std::vector<int> diskResult = hdd.readCombineData(combId, combFileId, file.minSizeId, file.maxSizeId);
        

        // TODO sort and use sorted result

        for(size_t idx = 0; idx < diskResult.size(); idx += __SIZE_RES_DISK){
            const int max_h = diskResult[idx + __RES_DISK_MAX_H];
            const int max_w = diskResult[idx + __RES_DISK_MAX_W];

            if(max_h >= start_h && max_w >= start_w){
                result.push_back(diskResult[idx + __RES_DISK_A_IDX]);
                result.push_back(diskResult[idx + __RES_DISK_B_IDX]);
            }
        }

        // size_t idx = 0;
        // while(idx < diskResult.size() && (diskResult[idx + __RES_DISK_MAX_H] < start_h || diskResult[idx + __RES_DISK_MAX_W] < start_w)){
        //     idx += __SIZE_RES_DISK;
        // }

        // while(idx < diskResult.size() && diskResult[idx + __RES_DISK_MAX_H] <= end_h){
        //     const int size_w = diskResult[idx + __RES_DISK_MAX_H];
        //     if(size_w >= start_w && size_w <= end_w){
        //         result.push_back(diskResult[idx + __RES_DISK_A_IDX]);
        //         result.push_back(diskResult[idx + __RES_DISK_B_IDX]);

        //         std::cout << "h: " << diskResult[idx + __RES_DISK_MAX_H] << ", h: " << diskResult[idx + __RES_DISK_MAX_W];
        //         std::cout << ", a: " << diskResult[idx + __RES_DISK_A_IDX] << ", b: " << diskResult[idx + __RES_DISK_B_IDX] << std::endl;
                
        //     }

        //     idx += __SIZE_RES_DISK;
        // }
    }


    std::cout << "getValidCombIdxFromComb layouts: " << (result.size() / 2) << ", combId: " << combId << std::endl << std::endl << std::endl;
    return result;
}

// TODO save fitness on file an ignore low fitness here
std::vector<int16_t> Search::getCombinations(const std::vector<int16_t>& a, const std::vector<int16_t>& b, const std::vector<int>& indexes, const std::vector<int>& conns, const int layout_a_size, const int layout_b_size, const int h, const int w, const int tolerance){
    std::vector<int16_t> result;
    std::vector<int16_t> layout_a(layout_a_size, 0);
    std::vector<int16_t> layout_b(layout_b_size, 0);
    std::vector<int16_t> original_layout_b(layout_b_size, 0);

    // std::cout << std::endl << std::endl << "getCombinations init" << std::endl;
    // std::vector<int16_t> layout(layout_a_size + layout_b_size, 0);

    const int start_h = h - tolerance;
    const int start_w = w - tolerance;
    const int end_h = h + tolerance;
    const int end_w = w + tolerance;

    const int max_size_diff = 200;
	std::vector<int> usedDiff(max_size_diff * max_size_diff, -1);

	// TODO check if int is sufficient for a_idx and b_idx
    int prev_a_idx = -1;
    int prev_b_idx = -1;
    for(size_t i = 0; i < indexes.size(); i+=2){
        const int a_idx = indexes[i];
        const int b_idx = indexes[i + 1];

        // if(a_idx == 4381 && b_idx == 92326){
        //     std::cout << "start" << std::endl;
        // }

        if(a_idx != prev_a_idx){
            for(int j = 0; j < layout_a_size; j++)
                layout_a[j] = a[a_idx + j];

            prev_a_idx = a_idx;
        }

        if(b_idx != prev_b_idx){
            for(int j = 0; j < layout_b_size; j++)
                original_layout_b[j] = b[b_idx + j];

            prev_b_idx = b_idx;
        }


        // cozinha    0,  0,  20, 25   
        // lavanderia 0,  25, 20, 40   
        // sala       20, 0,  50, 40   
        // corredor   50, 0,  95, 10   
        // quarto     50, 10, 80, 40   
        // banheiro   80, 10, 95, 40

        std::vector<int> test_a {80 - 80, 10 - 10, 95 - 80, 40 - 10, 
                                 50 - 80, 10 - 10, 80 - 80, 40 - 10, 
                                 50 - 80, 0  - 10, 95 - 80, 10 - 10, };

        // std::vector<int> test_b {20 - 20, 0  - 0, 50 - 20, 40 - 0, 
        //                           0 - 20, 0  - 0, 20 - 20, 25 - 0, 
        //                           0 - 20, 25 - 0, 20 - 20, 40 - 0, };

        // bool equal_a = true;
        // bool equal_b = true;

        // for(int k = 0; k < layout_a_size && equal_a; k++){
        //     if(layout_a[k] != test_a[k])
        //         equal_a = false;
        // }

        // for(int k = 0; k < layout_b_size && equal_b; k++){
        //     if(layout_b[k] != test_b[k])
        //         equal_b = false;
        // }
        // if(equal_a){
            // std::cout << "equal a. i: " << i << std::endl;

            // std::cout << "original a:" << std::endl;
            // for(int k = 0; k < layout_a_size; k+=4){
            //     for(int l = 0; l < 4; l++){
            //         std::cout << layout_a[k + l] << ", ";
            //     }
            //     std::cout << std::endl;
            // }
            // std::cout << std::endl;

            // std::cout << "original b:" << std::endl;
            // for(int k = 0; k < layout_b_size; k+=4){
            //     for(int l = 0; l < 4; l++){
            //         std::cout << original_layout_b[k + l] << ", ";
            //     }
            //     std::cout << std::endl;
            // }
            // std::cout << std::endl;
            // std::cout << std::endl;
            // std::cout << std::endl;
            // getchar();
        // }
        // else {
        //     continue;
        // }
        // if(i == 61086){
        //     std::cout << "original a:" << std::endl;
        //     for(int k = 0; k < layout_a_size; k+=4){
        //         for(int l = 0; l < 4; l++){
        //             std::cout << layout_a[k + l] << ", ";
        //         }
        //         std::cout << std::endl;
        //     }
        //     std::cout << std::endl;

        //     std::cout << "original b:" << std::endl;
        //     for(int k = 0; k < layout_b_size; k+=4){
        //         for(int l = 0; l < 4; l++){
        //             std::cout << original_layout_b[k + l] << ", ";
        //         }
        //         std::cout << std::endl;
        //     }
        //     std::cout << std::endl;
        // }

        //8, 10, x  9, 11

        std::fill(usedDiff.begin(),usedDiff.end(),-1);
        for(size_t j = 0; j < conns.size(); j++){
            const int conn_id = conns[j];

            int srcConn = (conn_id >> __COMBINE_CONN_SRC_X_SHIFT) & __COMBINE_CONN_BITS;
            int dstConn = (conn_id >> __COMBINE_CONN_DST_X_SHIFT) & __COMBINE_CONN_BITS;

            int src = layout_a[srcConn];
            int dst = original_layout_b[dstConn];
            const int diffX = src - dst;
            
            // if(a_idx == 4381 && b_idx == 92326 && srcConn == 8 && dstConn == 10){
            //     std::cout << "b" << std::endl;
            // }

            srcConn = (conn_id >> __COMBINE_CONN_SRC_Y_SHIFT) & __COMBINE_CONN_BITS;
            dstConn = (conn_id >> __COMBINE_CONN_DST_Y_SHIFT) & __COMBINE_CONN_BITS;
            src = layout_a[srcConn];
            dst = original_layout_b[dstConn];
            const int diffY = src - dst;

            if(usedDiff[(diffX * max_size_diff) + diffY] == 1)
                continue;
            
            // if(a_idx == 4381 && b_idx == 92326 && diffX == -80 && srcConn == 9 && dstConn == 9){
            //     std::cout << "c " << result.size() << ", layouts: " << result.size() / (layout_a_size + layout_b_size) << std::endl;
            // }

            // if(i == 61086 && j == 18){
            //     std::cout << "i: " << i << ", j: " << j << ", diffX: " << diffX << ", diffY: " << diffY << ", layout_a_size: " << layout_a_size << ", layout_b_size: " << layout_b_size << std::endl;
            // }
            for(int k = 0; k < layout_b_size; k+=2){

                // if(i == 61086 && j == 18){
                //     std::cout << "sum k: " << k << ", original k: " << original_layout_b[k] << ", sum: " << original_layout_b[k] + diffX << ", original k + 1: " << original_layout_b[k+ 1] << ", sum: " << original_layout_b[k + 1] + diffY << std::endl;
                // }

                layout_b[k] = original_layout_b[k] + diffX;
                layout_b[k + 1] = original_layout_b[k + 1] + diffY;
            }

            // if(i == 61086 && j == 18){
            //     std::cout << "b after sum:" << std::endl;
            //     for(int k = 0; k < layout_b_size; k+=4){
            //         for(int l = 0; l < 4; l++){
            //             std::cout << layout_b[k + l] << ", ";
            //         }
            //         std::cout << std::endl;
            //     }
            //     std::cout << std::endl;
            // }

            int minH = 5000, maxH = -5000;
            int minW = 5000, maxW = -5000;
            for(int k = 0; k < layout_a_size; k+=4){
                if(layout_a[k + __UP] < minH)
                    minH = layout_a[k + __UP];
                if(layout_a[k + __DOWN] > maxH)
                    maxH = layout_a[k + __DOWN];
                if(layout_a[k + __LEFT] < minW)
                    minW = layout_a[k + __LEFT];
                if(layout_a[k + __RIGHT] > maxW)
                    maxW = layout_a[k + __RIGHT];
            }

            for(int k = 0; k < layout_b_size; k+=4){
                if(layout_b[k + __UP] < minH)
                    minH = layout_b[k + __UP];
                if(layout_b[k + __DOWN] > maxH)
                    maxH = layout_b[k + __DOWN];
                if(layout_b[k + __LEFT] < minW)
                    minW = layout_b[k + __LEFT];
                if(layout_b[k + __RIGHT] > maxW)
                    maxW = layout_b[k + __RIGHT];
            }

            const int sizeH = maxH - minH;
            const int sizeW = maxW - minW;

            // if(i == 61086 && j == 18){
            //     std::cout << "sizeH: " << sizeH << ", sizeW: " << sizeW << ", start_h: " << start_h << ", end_h: " << end_h << ", start_w: " << start_w << ", end_w: " << end_w << std::endl;
            //     std::cout << "(sizeH < start_h): " << (sizeH < start_h) << ", (sizeH > end_h): " << (sizeH > end_h) << ", (sizeW < start_w): " << (sizeW < start_w) << ", (sizeW > end_w): " << (sizeW > end_w) << std::endl;
            // }
            if((sizeH < start_h || sizeH > end_h) || (sizeW < start_w || sizeW > end_w))
                continue;

            bool validLayout = true;
            for(int k = 0; k < __SIZE_A_LAYOUT && validLayout; k+=4){
                const int a_left = layout_a[k];
                const int a_up = layout_a[k + __UP];
                const int a_down = layout_a[k + __DOWN];
                const int a_right = layout_a[k + __RIGHT];

                for(int l = 0; l < __SIZE_B_LAYOUT && validLayout; l+=4){
                    const int b_left = layout_b[l];
                    const int b_up = layout_b[l + __UP];
                    const int b_down = layout_b[l + __DOWN];
                    const int b_right = layout_b[l + __RIGHT];

                    validLayout = !check_overlap(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right);
                    
                    // if(check_adjacency(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right)){
                        // connections[i/4] |= 1 << (j/4) + __COMBINE_N_A;
                        // connections[(j/4) + __COMBINE_N_A] |= 1 << (i/4); 
                    // }
                }
            }

            if(!validLayout)
                continue;

            usedDiff[(diffX * max_size_diff) + diffY] = 1;
            for(int k = 0; k < layout_a_size; k++){
                result.push_back(layout_a[k]);
            }

            for(int k = 0; k < layout_b_size; k++){
                result.push_back(layout_b[k]);
            }

            // std::cout << "a_idx: " << a_idx << ", b_idx: " << b_idx << std::endl;
            // std::cout << "img. i: " << i << ", j: " << j << ", diffX: " << diffX << ", diffY: " << diffY << std::endl;
            // std::cout << "maxH: " << maxH << ", minH: " << minH << ", maxW: " << maxW << ", minW: " << minW << std::endl;

            // // std::cout << "sizeH: " << sizeH << ", sizeW: " << sizeW << ", start_h: " << start_h << ", end_h: " << end_h << ", start_w: " << start_w << ", end_w: " << end_w << std::endl;
            // // std::cout << "(sizeH < start_h): " << (sizeH < start_h) << ", (sizeH > end_h): " << (sizeH > end_h) << ", (sizeW < start_w): " << (sizeW < start_w) << ", (sizeW > end_w): " << (sizeW > end_w) << std::endl;
            // std::cout << "final a:" << std::endl;
            // for(int k = 0; k < layout_a_size; k+=4){
            //     for(int l = 0; l < 4; l++){
            //         std::cout << layout_a[k + l] << ", ";
            //     }
            //     std::cout << std::endl;
            // }
            // std::cout << std::endl;

            // std::cout << "final b:" << std::endl;
            // for(int k = 0; k < layout_b_size; k+=4){
            //     for(int l = 0; l < 4; l++){
            //         std::cout << layout_b[k + l] << ", ";
            //     }
            //     std::cout << std::endl;
            // }
            // std::cout << std::endl;

            // std::cout << "orginal b:" << std::endl;
            // for(int k = 0; k < layout_b_size; k+=4){
            //     for(int l = 0; l < 4; l++){
            //         std::cout << original_layout_b[k + l] << ", ";
            //     }
            //     std::cout << std::endl;
            // }
            // std::cout << std::endl;
            // std::cout << std::endl;
            // std::cout << std::endl;

            // getchar();

        }

        // break;
    }

    std::cout << "combination layouts: " << (result.size() / (layout_a_size + layout_b_size)) << std::endl << std::endl;
    return result;
}

void Search::getLayouts(Storage hdd, const int h, const int w){
    // // std::map<int, std::vector<int>> result;
    
    std::string outputPath = hdd.getImagesPath();

    std::vector<int> combIds = hdd.getSavedCombinationsCombIds();
    for(int combId : combIds){
        const int a_idx = combId >> __COMBINE_NAME_ROOMS_ID_SHIFT;
        const int b_idx = combId & __COMBINE_NAME_ROOMS_ID_BYTES;

        const std::vector<RoomConfig> rooms_a = hdd.getConfigsById(a_idx);
        const std::vector<RoomConfig> rooms_b = hdd.getConfigsById(b_idx);
        const int size_a = rooms_a.size() * 4;
        const int size_b = rooms_b.size() * 4;
        const int roomsCount = rooms_a.size() + rooms_b.size();

        std::cout << "combId: " << combId << std::endl;
        std::cout << "a: " << std::endl;
        for(RoomConfig room : rooms_a){
            Log::print(room);
        }
        
        std::cout << std::endl << std::endl << "b: " << std::endl;
        for(RoomConfig room : rooms_b){
            Log::print(room);
        }

        const std::vector<int> conns = CombineHandler::createConns(rooms_a.size(), rooms_b.size());

        // TODO replace with combination files instead of core files
        std::vector<int> layout_a_files_ids = hdd.getSavedCoreFiles(a_idx);
        std::vector<int> layout_b_files_ids = hdd.getSavedCoreFiles(b_idx);

        for(int layout_a_file_id : layout_a_files_ids){
            std::vector<int16_t> layout_a = hdd.readCoreData(a_idx, layout_a_file_id);

            for(int layout_b_file_id : layout_b_files_ids){
                std::vector<int16_t> layout_b = hdd.readCoreData(b_idx, layout_b_file_id);  

                const int combFileId = (layout_a_file_id << __COMBINE_NAME_ROOMS_ID_SHIFT) | layout_b_file_id;
                std::vector<int> indexes = Search::getValidCombIdxFromComb(hdd, combId, combFileId, h, w, 0);
                std::vector<int16_t> pts = getCombinations(layout_a, layout_b, indexes, conns, size_a, size_b, h, w, 0);

                // // Viewer::showLayouts(pts, size_a + size_b);

                const std::string filename = std::to_string(combId) + "_" + std::to_string(combFileId) + "_" + std::to_string(h) + "_" + std::to_string(w);
                Viewer::saveLayoutsImages(pts, roomsCount, 0, outputPath, filename);
            }
        }
    }
}



// bool Search::CalculatePts(std::vector<int16_t>& ptsX, std::vector<int16_t>& ptsY, const std::vector<int16_t>& a, const std::vector<int16_t>& b, int a_offset, int b_offset, const int n_a, const int n_b, const int conn, const int diffH, const int diffW){
//     const int layout_size_a = (n_a * 4) + 1;
//     // const int layout_size_b = n_a * 4 + 1;
//     const int ptsPerLayout_a = n_a * 2;
//     const int ptsPerLayout_b = n_b * 2;

//     // a_offset *= layout_size_a;
//     // b_offset *= layout_size_b;
//     const int a_offset_end = a_offset + layout_size_a;
    
//     const int srcConn = conn & 0b11;
//     const int dstConn = (conn >> 2) & 0b11;

//     // std::cout << "a_offset: " << a_offset << ", b_offset: " << b_offset << ", n_a: " << n_a << ", n_b: " << n_b << ", conn: " << conn << std::endl;
//     // std::cout << "srcConn: " << srcConn << ", dstConn: " << dstConn << std::endl;

//     if(srcConn == dstConn){
//         return false;
//     }

// 	int minH = 5000, maxH = -5000;
// 	int minW = 5000, maxW = -5000;

//     for(int i = 0; i < ptsPerLayout_a; i++){
//         ptsX[i] = a[a_offset + (i * 2)];
//         ptsY[i] = a[a_offset + (i * 2) + 1];

//         if(ptsX[i] > maxW)
//             maxW = ptsX[i];
//         if(ptsX[i] < minW)
//             minW = ptsX[i];

//         if(ptsY[i] > maxH)
//             maxH = ptsY[i];
//         if(ptsY[i] < minH)
//             minH = ptsY[i];
//     }

//     // std::cout << "a: ";
//     // for(int i = 0; i < n_a * 4; i++){
//     //     std::cout << a[a_offset + i] << ", ";
//     // }


//     // std::cout << std::endl << "b: ";
//     // for(int i = 0; i < n_b * 4; i++){
//     //     std::cout << b[b_offset + i] << ", ";
//     // }
//     // std::cout << std::endl;
//     // std::cout << "offsetX: " << offsetX << ", offsetY: " << offsetY << std::endl;

//     int dstX = 0;
//     int dstY = 0;
//     if(dstConn == 0 || dstConn == 2)
//         dstX = b[b_offset];
//     else 
//         dstX = b[b_offset + 2];
        
//     if(dstConn == 0 || dstConn == 1)
//         dstY = b[b_offset + 1];
//     else 
//         dstY = b[b_offset + 3];

//     int srcX = 0;
//     int srcY = 0;
//     if(srcConn == 0 || srcConn == 2)
//         srcX = a[a_offset_end - 5];
//     else    
//         srcX = a[a_offset_end - 3];
        
//     if(srcConn == 0 || srcConn == 1)
//         srcY = a[a_offset_end - 4];
//     else 
//         srcY = a[a_offset_end - 2];

//     const int offsetX = srcX - dstX;
//     const int offsetY = srcY - dstY;
//     // std::cout << "srcX: " << srcX << ", dstX: " << dstX << ", srcY: " << srcY << ", dstY: " << dstY << std::endl;
//     // std::cout << "offsetX: " << offsetX << ", offsetY: " << offsetY << std::endl;
//     // std::cout << std::endl;

//     for(int i = 0; i < ptsPerLayout_b; i++){
//         const int idx = i + ptsPerLayout_a;
//         ptsX[idx] = b[b_offset + (i * 2)] + offsetX;
//         ptsY[idx] = b[b_offset + (i * 2) + 1] + offsetY;

//         if(ptsX[idx] > maxW)
//             maxW = ptsX[idx];
//         if(ptsX[idx] < minW)
//             minW = ptsX[idx];
            
//         if(ptsY[idx] > maxH)
//             maxH = ptsY[idx];
//         if(ptsY[idx] < minH)
//             minH = ptsY[idx];
//     }

//     // std::cout << "pts: ";
//     // for(int i = 0; i < ptsPerLayout_a + ptsPerLayout_b; i++){
//     //     std::cout << "(" << ptsX[i] << ", " << ptsY[i] << "), ";
//     // }
//     // std::cout << std::endl;
//     // std::cout << "maxW: " << maxW << ", minW: " << minW << ", maxH: " << maxH << ", minH: " << minH << std::endl;
//     // std::cout << "maxW - minW: " << maxW - minW << ", maxH - minH: " << maxH - minH << std::endl;

//     if(maxW - minW != diffW || maxH - minH != diffH)
//         return false;


//     for(int i = 0; i < n_a + n_b; i++){
// 		const int a_left = ptsX[i * 2];
// 		const int a_right = ptsX[(i * 2) + 1];

// 		const int a_up = ptsY[i * 2];
// 		const int a_down = ptsY[(i * 2) + 1];

//         for(int j = i + 1; j < n_a + n_b; j++){
//             const int b_left = ptsX[j * 2];
//             const int b_right = ptsX[(j * 2) + 1];
            
//             const int b_up = ptsY[j * 2];
//             const int b_down = ptsY[(j * 2) + 1];

//             // std::cout << "a_left: " << a_left << ", a_right: " << a_right << ", a_up: " << a_up << ", a_down: " << a_down << std::endl;
//             // std::cout << "b_left: " << b_left << ", b_right: " << b_right << ", b_up: " << b_up << ", b_down: " << b_down << std::endl;
//             // std::cout << "check_overlap: " << Search::check_overlap(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right) << std::endl;
//             // std::cout << "check_adjacency: " << Search::check_adjacency(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right) << std::endl;
//             // std::cout << std::endl;

//             if(Search::check_overlap(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right))
// 				return false;
			
// 			// if(Search::check_adjacency(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right))
//             //     return false;
//         }
//     }
//     // std::cout << std::endl;
//     // std::cout << "-----------------------------------";
//     // std::cout << std::endl;
//     // std::cout << std::endl;

//     return true;
// }

// void Search::ShowContent(const std::vector<int>& cudaResult, const std::vector<int16_t>& a, const std::vector<int16_t>& b, const int n_a, const int n_b, std::string imagesPath){
//     std::vector<int16_t> ptsX((n_a + n_b) * 2, 0);
//     std::vector<int16_t> ptsY((n_a + n_b) * 2, 0);

//     std::vector<int> last_i;
//     std::vector<int> last_j;

//     for(unsigned long i = 0; i < cudaResult.size(); i+= __SIZE_RES){
// 		int diffH = cudaResult[i];
// 		int diffW = cudaResult[i + 1];
// 		int a_layout_idx = cudaResult[i + 2];
// 		int b_layout_idx = cudaResult[i + 3];


//         // if(diffH != 40 || diffW != 95)
//         //     continue;

//         if(i < 472)
//             continue;

//         // for(int j = 0; j < __COMBINE_CONN; j++){
//         for(int j = 0; j < 8; j++){

//             const int conn = j + 1 + j/4;
//             std::cout << "1 i: " << i << ", j: " << j << ", conn: " << conn << std::endl;

//             // std::fill (ptsX.begin(), ptsX.end(), 0);
//             // std::fill (ptsY.begin(), ptsY.end(), 0);
//             if(Search::CalculatePts(ptsX, ptsY, a, b, a_layout_idx, b_layout_idx, n_a, n_b, conn, diffH, diffW)){
//                 // std::cout << std::endl;
//                 std::cout << "i: " << i << ", j: " << j << ", conn: " << conn << std::endl;
//                 std::cout << "diffH: " << diffH << ", diffW: " << diffW << ", a_layout_idx: " << a_layout_idx << ", b_layout_idx: " << b_layout_idx << std::endl;

//                 int dir = CVHelper::showLayoutMove(ptsX, ptsY);
//                 if(dir == -1 && last_i.size() == 0){
//                     j = -1;
//                 }
//                 else if(dir == -1){
//                     i = last_i.back(); last_i.pop_back(); 
//                     j = last_j.back() - 1; last_j.pop_back(); 

//                     diffH = cudaResult[i];
//                     diffW = cudaResult[i + 1];
//                     a_layout_idx = cudaResult[i + 2];
//                     b_layout_idx = cudaResult[i + 3];
//                 } else {
//                     last_i.push_back(i);
//                     last_j.push_back(j);
//                 }

//                 // std::cout << "3 i: " << i << ", j: " << j << ", conn: " << conn << std::endl;
//                 std::cout << std::endl;
//                 // std::cout << std::endl;

//                 if(false){
//                     std::string fullPath = imagesPath + "/" + std::to_string(diffH) + "_" + std::to_string(diffW) + "_" + std::to_string(a_layout_idx) + "_" + std::to_string(b_layout_idx) + "_" + std::to_string(j) + ".png";
//                     CVHelper::saveImage(ptsX, ptsY, fullPath);
//                 }
//             }
//             // std::cout << std::endl;
            
//             // CVHelper::showLayout(ptsX, ptsY);
//             // break;
//         }
//     }
// }