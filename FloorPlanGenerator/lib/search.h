#ifndef SEARCH
#define SEARCH
#include <vector>

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

    static bool CalculatePts(std::vector<int16_t>& ptsX, std::vector<int16_t>& ptsY, const std::vector<int16_t>& a, const std::vector<int16_t>& b, int a_offset, int b_offset, const int n_a, const int n_b, const int conn, const int diffH, const int diffW);
    static void ShowContent(const std::vector<int>& cudaResult, const std::vector<int16_t>& a, const std::vector<int16_t>& b, const int n_a, const int n_b);
};

#endif //SEARCH