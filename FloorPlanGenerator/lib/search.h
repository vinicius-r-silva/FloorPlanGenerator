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

public:
    /** 
     * @brief Search Constructor
     * @return None
    */
    Search();

    static bool check_adjacency(const int a_up, const int a_down, const int a_left, const int a_right, const int b_up, const int b_down, const int b_left, const int b_right);
    static bool check_overlap(const int a_up, const int a_down, const int a_left, const int a_right, const int b_up, const int b_down, const int b_left, const int b_right);


    static std::vector<int> getValidCombIdxFromComb(Storage hdd, const int combId, const int combFileId, const int h, const int w, const int tolerance);

    static void getLayouts(Storage hdd, const int h, const int w);

    static std::vector<int16_t> getCombinations(const std::vector<int16_t>& a, const std::vector<int16_t>& b, const std::vector<int>& indexes, const std::vector<int>& conns, const int layout_a_size, const int layout_b_size, const int h, const int w, const int tolerance);

    // static bool CalculatePts(std::vector<int16_t>& ptsX, std::vector<int16_t>& ptsY, const std::vector<int16_t>& a, const std::vector<int16_t>& b, int a_offset, int b_offset, const int n_a, const int n_b, const int conn, const int diffH, const int diffW);
    // static void ShowContent(const std::vector<int>& cudaResult, const std::vector<int16_t>& a, const std::vector<int16_t>& b, const int n_a, const int n_b, std::string imagesPath);
};

#endif //SEARCH