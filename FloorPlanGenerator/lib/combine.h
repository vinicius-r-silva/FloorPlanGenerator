#ifndef COMBINE
#define COMBINE
#include <vector>

/** 
 * @brief Combine existing connections
*/
class Combine
{

public:
    /** 
     * @brief Combine Constructor
     * @return None
    */
    Combine();


    static void getValidLayoutCombs(const std::vector<int16_t>& a, const std::vector<int16_t>& b, const int n_a, const int n_b);
};

#endif //COMBINE