#ifndef GENERATE
#define GENERATE

#include <globals.h>
#include <vector>

/** 
 * @brief Generate new connection values
*/
class Generate
{
private:
    inline static bool check_overlap(int a_left, int a_right, int a_up, int a_down, int b_left, int b_right, int b_up, int b_down);


public:
    /** 
     * @brief Generate Constructor
     * @return None
    */
    Generate();

    /*!
        @brief Given a vector of RoomConfig setups, iterate over every possible room sizes
        @param[in] rooms vector containg all rooms informations, such as minimum and maximum sizes
        @return vector of vector of vector of layout combination. result[a][b][c] = d, a -> room size id, b -> permutation id, d -> layout points
    */
    static std::vector<std::vector<std::vector<int16_t>>> SizeLoop(const std::vector<RoomConfig>& rooms, const std::vector<int>& adjValues);

    /*!
        @brief Iterate over every possible connection between the given rooms 
        @param[in] order, specify the order of the rooms to connect
        @param[in] sizeH Height value of each room setup
        @param[in] sizeW Width value of each room setup
        @param[in] n     number of rooms
        @param[in] NConn Number of possible connections
        @return vector with layout points for every successful connection (n*4 int per layout)
    */
    static std::vector<int16_t> ConnLoop(const std::vector<int>& order, const int16_t *sizeH, const int16_t *sizeW, const int n, const int NConn, const std::vector<int>& adjValues, std::vector<int16_t> adjId);


    /*!
        @brief Iterate over every room permutation
        @param[in] sizeH Height value of each room setup
        @param[in] sizeW Width value of each room setup
        @param[in] n     number of rooms
        @return  vector of vector of layout combination. result[a][b] = c, a -> permutation id, c -> layout points
    */
    static std::vector<std::vector<int16_t>> roomPerm(const int16_t *sizeH, const int16_t *sizeW, const int n, const std::vector<int>& adjValues, std::vector<int16_t> adjId);

};

#endif //GENERATE