#include "../lib/search.h"
#include "../lib/cvHelper.h"
#include "../lib/globals.h"
#include <iostream>
#include <string>
#include <stdlib.h>
#include "../cuda/combineHandler.h"
#include "../lib/viewer.h"
#include "../lib/log.h"
#include "../lib/calculator.h"

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

std::vector<int> Search::getValidCombIdxFromComb(Storage hdd, const int combId, const int combFileId, const int minDiffH, const int minDiffW){
    std::vector<int> result;

    // const int start_h = h - tolerance;
    // const int start_w = w - tolerance;
    // const int end_h = h + tolerance;
    // const int end_w = w + tolerance;

    std::vector<CombinationResult> resultFiles = hdd.getSavedCombinations(combId, combFileId);
    std::cout << "getValidCombIdxFromComb combId: " << combId << ", combFileId: " << combFileId << std::endl;

    for(CombinationResult file : resultFiles){
        // int file_min_h = file.minSizeId >> __RES_FILE_LENGHT_BITS;
        // int file_min_w = file.minSizeId & __RES_FILE_LENGHT_AND_RULE;

        int file_max_h = file.maxSizeId >> __RES_FILE_LENGHT_BITS;
        int file_max_w = file.maxSizeId & __RES_FILE_LENGHT_AND_RULE;

        if(file_max_h < minDiffH || file_max_w < minDiffW)
            continue;

        std::vector<int> diskResult = hdd.readCombineData(combId, combFileId, file.minSizeId, file.maxSizeId);
        

        // TODO sort and use sorted result

        for(size_t idx = 0; idx < diskResult.size(); idx += __SIZE_RES_DISK){
            const int max_h = diskResult[idx + __RES_DISK_MAX_H];
            const int max_w = diskResult[idx + __RES_DISK_MAX_W];

            if(max_h >= minDiffH && max_w >= minDiffW){
                result.push_back(diskResult[idx + __RES_DISK_A_IDX]);
                result.push_back(diskResult[idx + __RES_DISK_B_IDX]);
            }
        }
    }


    std::cout << "getValidCombIdxFromComb layouts: " << (result.size() / 2) << ", combId: " << combId << std::endl << std::endl << std::endl;
    return result;
}

// TODO save fitness on file an ignore low fitness here
std::vector<int16_t> Search::getCombinations(
    const std::vector<int16_t>& inputShape,
    const std::vector<int16_t>& a, 
    const std::vector<int16_t>& b, 
    const std::vector<int>& indexes, 
    const std::vector<int>& conns, 
    const std::vector<int>& req_adj, 
    const int layout_a_pts, 
    const int layout_b_pts, 
    const int minDiffH,
    const int minDiffW,
    const int maxDiffH,
    const int maxDiffW,
    const int minArea,
    const int maxArea)
{
    std::vector<int16_t> result;
    std::vector<int16_t> layout_a(layout_a_pts, 0);
    std::vector<int16_t> layout_b(layout_b_pts, 0);
    std::vector<int16_t> original_layout_b(layout_b_pts, 0);

    int a_rid_idx = -1;
    int b_rid_idx = -1;
    const int layout_input_size = inputShape.size();

    const int n_a = layout_a_pts / 4;
    const int n_b = layout_b_pts / 4;

    const int max_size_diff = 400;
    const int max_size_diff_offset = max_size_diff / 2;

    int layout_a_area = 0;
    int layout_b_area = 0;
    int intersectionArea = 0;

    std::vector<int> adj(__SIZE_ADJ_TYPES, 0);
    std::vector<int> adj_count(__SIZE_ADJ_TYPES, 0);
	std::vector<int> usedDiff(max_size_diff * max_size_diff, -1);
	std::vector<int> connections(n_a + n_b, 0);

	// TODO check if int is sufficient for a_idx and b_idx
    int prev_a_idx = -1;
    int prev_b_idx = -1;
    int center_h_a = 0;
    int center_w_a = 0;
    int center_h_b = 0;
    int center_w_b = 0;
    for(size_t i = 0; i < indexes.size(); i+=2){
        const int a_idx = indexes[i];
        const int b_idx = indexes[i + 1]; 

        std::fill(usedDiff.begin(),usedDiff.end(),-1);  


        if(a_idx != prev_a_idx){
            layout_a_area = 0;
            center_h_a = 0;
            center_w_a = 0;

            for(int j = 0; j < layout_a_pts; j+=4){
                const int offset = a_idx + j;
                const int left = a[offset + __LEFT];
                const int up = a[offset + __UP];
                const int right = a[offset + __RIGHT];
                const int down = a[offset + __DOWN];

                const int roomArea = (down - up) * (right - left);

                center_h_a += 5 * roomArea * (down + up);
                center_w_a += 5 * roomArea * (right + left);

                layout_a_area += roomArea;
                layout_a[j + __LEFT] = left;
                layout_a[j + __UP] = up;
                layout_a[j + __RIGHT] = right;
                layout_a[j + __DOWN] = down;
            }

            a_rid_idx = a[a_idx + layout_a_pts];
            prev_a_idx = a_idx;
        }

        if(b_idx != prev_b_idx){
            layout_b_area = 0;
            center_h_b = 0;
            center_w_b = 0;

            for(int j = 0; j < layout_b_pts; j+=4){
                const int offset = b_idx + j;
                const int left = b[offset + __LEFT];
                const int up = b[offset + __UP];
                const int right = b[offset + __RIGHT];
                const int down = b[offset + __DOWN];

                const int roomArea = (down - up) * (right - left);

                center_h_b += 5 * roomArea * (down + up);
                center_w_b += 5 * roomArea * (right + left);

                layout_b_area += roomArea;
                original_layout_b[j + __LEFT] = left;
                original_layout_b[j + __UP] = up;
                original_layout_b[j + __RIGHT] = right;
                original_layout_b[j + __DOWN] = down;
            }

            b_rid_idx = b[b_idx + layout_b_pts];
            prev_b_idx = b_idx;
        }

        const int totalArea = layout_a_area + layout_b_area;
        if(totalArea < minArea || totalArea > maxArea)
            continue;

        for(size_t j = 0; j < conns.size(); j++){
            // if(i == 0 && j == 14)
            //     std::cout << "init" << std::endl;
            const int conn_id = conns[j];

            int srcConn = (conn_id >> __COMBINE_CONN_SRC_X_SHIFT) & __COMBINE_CONN_BITS;
            int dstConn = (conn_id >> __COMBINE_CONN_DST_X_SHIFT) & __COMBINE_CONN_BITS;

            int src = layout_a[srcConn];
            int dst = original_layout_b[dstConn];
            const int diffX = src - dst;

            srcConn = (conn_id >> __COMBINE_CONN_SRC_Y_SHIFT) & __COMBINE_CONN_BITS;
            dstConn = (conn_id >> __COMBINE_CONN_DST_Y_SHIFT) & __COMBINE_CONN_BITS;
            src = layout_a[srcConn];
            dst = original_layout_b[dstConn];
            const int diffY = src - dst;

            const int usedDiffIdx = ((diffX + max_size_diff_offset) * max_size_diff) + (diffY + max_size_diff_offset);
            if(usedDiff[usedDiffIdx] == 1)
                continue;
        
            for(int k = 0; k < layout_b_pts; k+=2){
                layout_b[k] = original_layout_b[k] + diffX;
                layout_b[k + 1] = original_layout_b[k + 1] + diffY;
            }

            int minH = 5000, maxH = -5000;
            int minW = 5000, maxW = -5000;
            for(int k = 0; k < layout_a_pts; k+=4){
                if(layout_a[k + __UP] < minH)
                    minH = layout_a[k + __UP];
                if(layout_a[k + __DOWN] > maxH)
                    maxH = layout_a[k + __DOWN];
                if(layout_a[k + __LEFT] < minW)
                    minW = layout_a[k + __LEFT];
                if(layout_a[k + __RIGHT] > maxW)
                    maxW = layout_a[k + __RIGHT];
            }

            for(int k = 0; k < layout_b_pts; k+=4){
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

            if(sizeH < minDiffH || sizeW < minDiffW || sizeH > maxDiffH || sizeW > maxDiffW)
                continue;

            bool validLayout = true;
            std::fill(connections.begin(),connections.end(),0);
            for(int k = 0; k < layout_a_pts && validLayout; k+=4){
                const int a_left = layout_a[k + __LEFT];
                const int a_up = layout_a[k + __UP];
                const int a_right = layout_a[k + __RIGHT];
                const int a_down = layout_a[k + __DOWN];

                for(int l = 0; l < layout_b_pts && validLayout; l+=4){
                    const int b_left = layout_b[l + __LEFT];
                    const int b_up = layout_b[l + __UP];
                    const int b_right = layout_b[l + __RIGHT];
                    const int b_down = layout_b[l + __DOWN];

                    validLayout = !check_overlap(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right);
                    
                    if(check_adjacency(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right)){
                        connections[k/4] |= 1 << ((l/4) + n_a);
                        connections[(l/4) + n_a] |= 1 << (k/4); 
                    }
                }
            }

            if(!validLayout)
                continue;

            for(int k = 0; k < layout_a_pts && validLayout; k+=4){
                const int a_left = layout_a[k + __LEFT];
                const int a_up = layout_a[k + __UP];
                const int a_right = layout_a[k + __RIGHT];
                const int a_down = layout_a[k + __DOWN];
                // layoutArea += (a_down - a_up) * (a_right - a_left);

                for(int l = k + 4; l < layout_a_pts && validLayout; l+=4){
                    const int b_left = layout_a[l + __LEFT];
                    const int b_up = layout_a[l + __UP];
                    const int b_right = layout_a[l + __RIGHT];
                    const int b_down = layout_a[l + __DOWN];
                    
                    if(check_adjacency(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right)){
                        connections[k/4] |= 1 << (l/4);
                        connections[l/4] |= 1 << (k/4); 
                    }
                }
            }

            for(int k = 0; k < layout_b_pts && validLayout; k+=4){
                const int a_left = layout_b[k + __LEFT];
                const int a_up = layout_b[k + __UP];
                const int a_right = layout_b[k + __RIGHT];
                const int a_down = layout_b[k + __DOWN];
                // layoutArea += (a_down - a_up) * (a_right - a_left);

                for(int l = k + 4; l < layout_b_pts && validLayout; l+=4){
                    const int b_left = layout_b[l + __LEFT];
                    const int b_up = layout_b[l + __UP];
                    const int b_right = layout_b[l + __RIGHT];
                    const int b_down = layout_b[l + __DOWN];
                    
                    if(check_adjacency(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right)){
                        connections[(k/4) + n_a] |= 1 << ((l/4) + n_a);
                        connections[(l/4) + n_a] |= 1 << ((k/4) + n_a); 
                    }
                }
            }
                
            std::fill(adj.begin(),adj.end(),0);  
            std::fill(adj_count.begin(),adj_count.end(),0);  

            for(int k = 0; k < n_a; k++){
                const int rplannyId = (a_rid_idx >> (k * __RID_BITS_SIZE)) & __RID_BITS;
                adj_count[rplannyId] |= 1 << k;
                adj[rplannyId] |= connections[k];
            }
            
            for(int k = 0; k < n_b; k++){
                const int rplannyId = (b_rid_idx >> (k * __RID_BITS_SIZE)) & __RID_BITS;
                adj_count[rplannyId] |= 1 << (k + n_a);
                adj[rplannyId] |= connections[k + n_a];
            }

            for(int k = 0; k < __SIZE_ADJ_TYPES && validLayout; k++){
                for(int l = 0; l < __SIZE_ADJ_TYPES && validLayout; l++){
                    const int req_adj_idx = (k * __SIZE_ADJ_TYPES) + l;
                    if(req_adj[req_adj_idx] == REQ_ANY && !(adj[l] & adj_count[k]))
                        validLayout = false;

                    if(req_adj[req_adj_idx] == REQ_ALL && (adj[l] & adj_count[k]) != adj_count[k])
                        validLayout = false;
                }
            }

            if(!validLayout)
                continue;

            for(int k = 0; k < n_a + n_b; k++){
                const int conns = connections[k];
                for(int l = k + 1; l < n_a + n_b; l++){
                    if(connections[l] & (1 << k))
                        connections[l] |= conns;
                }
            }

            // TODO calculate __CONN_CHECK_IDX and __CONN_CHECK

            if(connections[__CONN_CHECK_IDX] != __CONN_CHECK)
                continue;

            const int center_h = (center_h_a + center_h_b + (diffY * layout_b_area * 10)) / (totalArea * 10); 
            const int center_w = (center_w_a + center_w_b + (diffX * layout_b_area * 10)) / (totalArea * 10);
            intersectionArea = 0;

            for(int k = 0; k < layout_input_size; k+=4){
                const int input_left = inputShape[k + __LEFT];
                const int input_up = inputShape[k + __UP];
                const int input_right = inputShape[k + __RIGHT];
                const int input_down = inputShape[k + __DOWN];

                for(int l = 0; l < layout_a_pts; l+=4){
                    const int a_left = layout_a[l + __LEFT] - center_w;
                    const int a_up = layout_a[l + __UP] - center_h;
                    const int a_right = layout_a[l + __RIGHT] - center_w;
                    const int a_down = layout_a[l + __DOWN] - center_h;

                    intersectionArea += Calculator::IntersectionArea(input_up, input_down, input_left, input_right, a_up, a_down, a_left, a_right);
                }

                for(int l = 0; l < layout_b_pts; l+=4){
                    const int b_left = layout_b[l + __LEFT] - center_w;
                    const int b_up = layout_b[l + __UP] - center_h;
                    const int b_right = layout_b[l + __RIGHT] - center_w;
                    const int b_down = layout_b[l + __DOWN] - center_h;

                    intersectionArea += Calculator::IntersectionArea(input_up, input_down, input_left, input_right, b_up, b_down, b_left, b_right);
                }
            }

            if(intersectionArea < minArea)
                continue;

            usedDiff[usedDiffIdx] = 1;
            for(int k = 0; k < layout_a_pts; k++){
                result.push_back(layout_a[k]);
            }

            for(int k = 0; k < layout_b_pts; k++){
                result.push_back(layout_b[k]);
            }
            result.push_back(std::abs(intersectionArea - totalArea));
        }

        // break;
    }

    std::cout << "combination layouts: " << (result.size() / (layout_a_pts + layout_b_pts)) << std::endl << std::endl;
    return result;
}

// void Search::getLayouts(Storage hdd, const int h, const int w, const std::vector<int16_t>& inputShape){

void Search::getLayouts(const std::vector<int16_t>& inputShape, Storage hdd){
    // // std::map<int, std::vector<int>> result;
    // moveToCenterOfMass(inputShape);

    int inputArea = 0;
	int minH = 5000, maxH = -5000;
	int minW = 5000, maxW = -5000;
	for(size_t i = 0; i < inputShape.size(); i+=4){
        int left = inputShape[i + __LEFT];
        int up = inputShape[i + __UP];
        int down = inputShape[i + __DOWN];
        int right = inputShape[i + __RIGHT];

        inputArea += (down - up) * (right - left);

        minH = std::min(minH, up);
        maxH = std::max(maxH, down);
        minW = std::min(minW, left);
        maxW = std::max(maxW, right);
	}

    // const int diffH = maxH - minH;
    // const int diffW = maxW - minW;

    // TODO get maximumBoundingBoxDimensions imaging an wall of 1meter extending outisede the desires box, how much would have do increase to be no longer valid?
    // instead of 1meter, maybe 1 quarter of the box side size

    std::pair<int, int> minimumBoundingBoxDimensions = Calculator::getMinimumAcceptableBoundingBoxDimensions(inputShape);
    const int minDiffH = minimumBoundingBoxDimensions.first;
    const int minDiffW = minimumBoundingBoxDimensions.second;

    std::pair<int, int> maximumBoundingBoxDimensions = Calculator::getMaximumAcceptableBoundingBoxDimensions(inputShape);
    const int maxDiffH = maximumBoundingBoxDimensions.first;
    const int maxDiffW = maximumBoundingBoxDimensions.second;
    const int minArea = inputArea - ((inputArea * __SEARCH_TOLERANCE_AREA_PCT) / 100);
    const int maxArea = inputArea + ((inputArea * __SEARCH_TOLERANCE_AREA_PCT) / 100);
    
    std::string outputPath = hdd.getImagesPath();

    std::vector<int> allReq = hdd.getReqAdjValues();
    std::vector<int> combIds = hdd.getSavedCombinationsCombIds();
    for(int combId : combIds){
        // if(combId != 917521)
        //     continue;

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
                std::vector<int> indexes = Search::getValidCombIdxFromComb(hdd, combId, combFileId, minDiffH, minDiffW);
                std::vector<int16_t> pts = getCombinations(inputShape, layout_a, layout_b, indexes, conns, allReq, size_a, size_b, minDiffH, minDiffW, maxDiffH, maxDiffW, minArea, maxArea);

                // // Viewer::showLayouts(pts, size_a + size_b);
                const std::string filename = std::to_string(combId) + "_" + std::to_string(combFileId);
                // pts.resize(200);

                Viewer::saveLayoutsImages(pts, roomsCount, 1, outputPath, filename);
                return;
            }
        }
    }
}

void Search::moveToCenterOfMass(std::vector<int16_t>& layout){
    std::pair<int, int> centerOfMass = Calculator::getCenterOfMass(layout);

    const int centerH = centerOfMass.first;
    const int centerW = centerOfMass.second;

	for(size_t i = 0; i < layout.size(); i+=2){
        layout[i] -= centerW;
        layout[i + 1] -= centerH;
    }
}