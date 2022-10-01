#include <iostream>
#include <vector>
#include <algorithm>
#include "../lib/calculator.h"
#include "../lib/globals.h"

/** 
 * @brief Calculator Constructor
 * @return None
*/
Calculator::Calculator(){
}
    

/*!
    @brief Factorial Calculator (n!)
    @param[in] x input to calculate the factorial
    @return (int) factorial of x
*/
int Calculator::Factorial(int x){
    int res = 1;
    for(; x > 1; x--)
        res *= x;

    return res;
}


/*!
    @brief Calculates the number of possible connections given the quantity of rooms
    @param[in] n input to calculate the number of connections
    @return (int) number of connections
*/
int Calculator::NConnections(int n){
    int res = 1;
    for(; n > 1; n--)
        res *= 16;

    return res;
}