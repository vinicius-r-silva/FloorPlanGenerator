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
    
    static int NRoomSizes(const std::vector<RoomConfig>& rooms);
    static int NCombination(int k, int n);
    static void totalOfCombinations(const std::vector<RoomConfig>& setups, const int n);
};

#endif //CALCUALTOR