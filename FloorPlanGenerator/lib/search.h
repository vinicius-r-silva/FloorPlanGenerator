#ifndef SEARCH
#define SEARCH
#include <vector>
#include <string>
#include "storage.h"

/** 
 * @brief Search existing connections
*/
class Search
{
private:
    static bool check_adjacency(const int a_up, const int a_down, const int a_left, const int a_right, const int b_up, const int b_down, const int b_left, const int b_right);
    
    static bool check_overlap(const int a_up, const int a_down, const int a_left, const int a_right, const int b_up, const int b_down, const int b_left, const int b_right);

    static std::vector<int> getValidCombIdxFromComb(Storage hdd, const int combId, const int combFileId, const int minDiffH, const int minDiffW);

    static std::vector<int16_t> getCombinations(
        const std::vector<int16_t>& inputShape,
        const std::vector<int16_t>& a, 
        const std::vector<int16_t>& b, 
        const std::vector<int>& indexes, 
        const std::vector<int>& conns, 
        const std::vector<int>& req_adj, 
        const int layout_a_size, 
        const int layout_b_size, 
        const int minDiffH,
        const int minDiffW,
        const int maxDiffH,
        const int maxDiffW,
        const int minArea,
        const int maxArea);

public:
    /** 
     * @brief Search Constructor
     * @return None
    */
    Search();

    static void getLayouts(const std::vector<int16_t>& inputShape, Storage hdd);

    static void moveToCenterOfMass(std::vector<int16_t>& layout);
};

#endif //SEARCH