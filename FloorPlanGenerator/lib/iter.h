#ifndef COMBINATION
#define COMBINATION

#include <globals.h>
#include <vector>


class PermLoopRes {
    public:
        // PermLoopRes() = default;
        std::vector<std::vector<int>> perms;
        std::vector<std::vector<int>> conns;
};

class SizeLoopRes {
    public:
        // PermLoopRes() = default;
        std::vector<int> roomsId;
        std::vector<PermLoopRes> perms;
};

/** 
 * @brief Handles all console's read/write
*/
class Iter
{
private:
    /*!
        @brief Get the next combination of k elements in a vector of size n
        @details https://stackoverflow.com/questions/5095407/all-combinations-of-k-elements-out-of-n
        @param[in] first begin of the vector
        @param[in] k current position 
        @param[in] last end of the vector
        @return True if there is a new combination, false otherwise
    */
    template <typename Iterator>
    inline static bool next_combination(const Iterator first, Iterator k, const Iterator last);
    inline static bool check_overlap(int a_left, int a_right, int a_up, int a_down, int b_left, int b_right, int b_up, int b_down);


public:
    /** 
     * @brief Iter Constructor
     * @return None
    */
    Iter();

    /*!
        @brief Get all possible combinations of k elements in a vector of size n
        @details https://stackoverflow.com/questions/5095407/all-combinations-of-k-elements-out-of-n
        @param[in] setups vector containg all elements
        @param[in] k size of the combinations
        @return (vector of vector of RoomConfig) return a vector with all possible combinations where wich combination is a vector of RoomConfig 
    */
    static std::vector<std::vector<RoomConfig>> getAllComb(std::vector<RoomConfig> setups, int k);

    /*!
        @brief Calculate a new room's width and height
        @param[in] rooms vector containg all rooms informations, such as minimum and maximum sizes
        @param[out] sizeH rooms Height size
        @param[out] sizeW rooms Width size
        @return True if there is a next room size iteration, false otherwise
    */
    static bool nextRoomSize(std::vector<RoomConfig> rooms, int *sizeH, int *sizeW);

    /*!
        @brief Given a vector of RoomConfig setups, iterate over every possible room sizes
        @param[in] rooms vector containg all rooms informations, such as minimum and maximum sizes
        @return None
    */
    static SizeLoopRes SizeLoop(const std::vector<RoomConfig>& rooms);

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
        @return None
    */
    static PermLoopRes roomPerm(const int *sizeH, const int *sizeW, const int n);

};

#endif //COMBINATION