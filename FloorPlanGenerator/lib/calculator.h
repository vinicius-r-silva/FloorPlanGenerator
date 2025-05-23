#ifndef CALCUALTOR
#define CALCUALTOR

#include <globals.h>

/** 
 * @brief Handles all console's read/write
*/
class Calculator
{
private:


public:
    /** 
     * @brief Calculator Constructor
     * @return None
    */
    Calculator();

    /*!
        @brief Factorial Calculator (n!)
        @param[in] x input to calculate the factorial
        @return (int) factorial of x
    */
    static int Factorial(int x);


    static int NRotations(int n);

    /*!
        @brief Calculates the number of possible connections given the quantity of rooms
        @param[in] n input to calculate the number of connections
        @return (int) number of connections
    */
    static int NConnections(int n);
    

    /*!
        @brief Calculates the number of possible connections given the quantity of rooms but removes combinations that are garantee to overlap
        @param[in] n input to calculate the number of connections
        @return (int) number of connections
    */
    static int NConnectionsReduced(int n);
    

    /*!
        @brief Calculates the number of possible room's size combinations
        @param[in] rooms vector with each room information
        @return (int) number of room's size combinations
    */
    static int NRoomSizes(const std::vector<RoomConfig>& rooms);

    static int IntersectionArea(
        const int a_up, const int a_down, const int a_left, const int a_right, 
        const int b_up, const int b_down, const int b_left, const int b_right);

    /*!
        @brief Calculates the size of the upper (or lower) half of a matrix of size n
        @param[in] n size of matrix
        @return (int) lenght of upper half of matrix
    */
    static int lenghtHalfMatrix(const int n);
    

    /*!
        @brief Calculates the number of possible combination of n between k
        @param[in] k number of the total of elements
        @param[in] n number of the total of elements to select
        @return (int) number of combination
    */
    static int NCombination(int k, int n);
    

    /*!
        @brief Calculates the total number of possible layouts considering combination, rooms sizes, etc
        @param[in] setups every room configuration
        @param[in] n number of rooms per layout
        @return (int) number of possible layouts
    */
    static void totalOfCombinations(const std::vector<RoomConfig>& setups, const int n);

    static std::pair<int, int> getMinimumAcceptableBoundingBoxDimensions(const std::vector<int16_t>& input);

    static std::pair<int, int> getMaximumAcceptableBoundingBoxDimensions(const std::vector<int16_t>& input);

    static std::pair<int, int> getCenterOfMass(const std::vector<int16_t>& layout);
};

#endif //CALCUALTOR