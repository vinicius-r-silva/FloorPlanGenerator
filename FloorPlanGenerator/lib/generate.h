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
        @return vector of vector of vector of layout combination. result[a][b][c] = d, a -> room size id, b -> permutation id, d -> connection id
    */
    static std::vector<std::vector<std::vector<int>>> SizeLoop(const std::vector<RoomConfig>& rooms);

    /*!
        @brief Iterate over every possible connection between the given rooms 
        @param[in] order, specify the order of the rooms to connect
        @param[in] sizeH Height value of each room setup
        @param[in] sizeW Width value of each room setup
        @param[in] n     number of rooms
        @param[in] NConn Number of possible connections
        @return vector of every successful connection (int)
    */
    static std::vector<int> ConnLoop(const std::vector<int>& order, const int *sizeH, const int *sizeW, const int n, const int NConn);


    /*!
        @brief Iterate over every room permutation
        @param[in] sizeH Height value of each room setup
        @param[in] sizeW Width value of each room setup
        @param[in] n     number of rooms
        @return  vector of vector of layout combination. result[a][b] = c, a -> permutation id, c -> connection id
    */
    static std::vector<std::vector<int>> roomPerm(const int *sizeH, const int *sizeW, const int n);

};

#endif //GENERATE