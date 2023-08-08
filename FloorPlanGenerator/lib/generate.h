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
    /*!
        @brief Given two squares, returns if there is a overleap between the two
        @param[in] a_left   left side of square A (smallest value of the x axis)
        @param[in] a_right  right side of square A (biggest value of the x axis)
        @param[in] a_up     up side of square A (smallest value of the y axis)
        @param[in] a_down   down side of square A (biggest value of the y axis)
        @param[in] b_left   left side of square B (smallest value of the x axis)
        @param[in] b_right  right side of square B (biggest value of the x axis)
        @param[in] b_up     up side of square B (smallest value of the y axis)
        @param[in] b_down   down side of square B (biggest value of the y axis)
        @return (bool) true if there is a overleap, false otherwise
    */
    inline static bool check_overlap(int a_left, int a_right, int a_up, int a_down, int b_left, int b_right, int b_up, int b_down);

    /*!
        @brief Given two squares, returns true if one square touchs another
        @param[in] a_left   left side of square A (smallest value of the x axis)
        @param[in] a_right  right side of square A (biggest value of the x axis)
        @param[in] a_up     up side of square A (smallest value of the y axis)
        @param[in] a_down   down side of square A (biggest value of the y axis)
        @param[in] b_left   left side of square B (smallest value of the x axis)
        @param[in] b_right  right side of square B (biggest value of the x axis)
        @param[in] b_up     up side of square B (smallest value of the y axis)
        @param[in] b_down   down side of square B (biggest value of the y axis)
        @return (bool) true share of a same edge (even if it is partially), false otherwise
    */
    inline static bool check_adjacency(int a_left, int a_right, int a_up, int a_down, int b_left, int b_right, int b_up, int b_down); 


public:
    /** 
     * @brief Generated Constructor
     * @return None
    */
    Generate();

    /*!
        @brief Given a vector of RoomConfig setups, iterate over every possible room sizes
        @param[in] reqSize lengh of required matrix
        @param[in] allReq required rooms ajacency, used to force room adjacency in layout, such as a master room has to have a connection with a bathroom
        @param[in] allReqCount required rooms ajacency count of how many rules are related to each room class
        @param[in] rooms vector containg all rooms informations, such as minimum and maximum sizes
        @return vector of coordinates points. Every two points combines into a coordinate and every n * 4 coordinates makes a layout
    */
    static std::vector<int16_t> SizeLoop(
        const int reqSize,
        std::vector<int> allReq,
        std::vector<int> allReqCount,
        const std::vector<RoomConfig>& rooms);


    /*!
        @brief Iterate over every possible connection between the given rooms 
        @param[in] n     number of rooms
        @param[in] NConn Number of possible connections
        @param[in] reqSize lengh of required matrix
        @param[in] permIter iteration count of the permutation
        @param[in] sizeH Height value of each room setup
        @param[in] sizeW Width value of each room setup
        @param[in] order, specify the order of the rooms to connect
        @param[in] result, vector of points. Every two points combines into a coordinate and every n * 4 coordinates makes a layout
        @param[in] reqAdj required rooms ajacency, used to force room adjacency in layout, such as a master room has to have a connection with a bathroom
        @param[in] rooms vector containg all rooms informations, such as minimum and maximum sizes
        @return None. It changes the result array by pushing back layouts coordinates
    */
    static void ConnLoop(
        const int n, 
        const int NConn, 
        const int reqSize,
        const int permIter,
        const int16_t *sizeH, 
        const int16_t *sizeW, 
        std::vector<int16_t>& result,
        const std::vector<int>& order, 
        const std::vector<int>& reqAdj,
        const std::vector<RoomConfig>& rooms);


    /*!
        @brief Iterate over every room permutation
        @param[in] n number of rooms
        @param[in] NConn Number of possible connections
        @param[in] reqSize lengh of required matrix
        @param[in] sizeH Height value of each room setup
        @param[in] sizeW Width value of each room setup
        @param[in] result, vector of points. Every two points combines into a coordinate and every n * 4 coordinates makes a layout
        @param[in] reqAdj required rooms ajacency, used to force room adjacency in layout, such as a master room has to have a connection with a bathroom
        @param[in] rooms vector containg all rooms informations, such as minimum and maximum sizes
        @return None. It changes the result array by pushing back layouts coordinates
    */
    static void roomPerm(
    const int n, 
    const int NConn,
    const int reqSize,
    const int16_t *sizeH, 
    const int16_t *sizeW, 
    std::vector<int16_t>& result,
    const std::vector<int>& reqAdj,
    const std::vector<RoomConfig>& rooms);

};

#endif //GENERATE