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
};

#endif //CALCUALTOR