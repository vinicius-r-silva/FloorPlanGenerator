#include <iostream>
#include <vector>
#include <algorithm>
#include "../lib/calculator.h"
#include "../lib/globals.h"
#include "../lib/iter.h"

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
    @details f(2) = 16 || f(n) = f(n-1) * (n - 1) * 16
    @param[in] n input to calculate the number of connections
    @return (int) number of connections
*/
int Calculator::NConnections(int n){
    if(n == 1)
        return 0;

    int res = 1;
    for(int i = 1; i < n; i++){
        res *= i * 16;
    }

    return res;
}
    

/*!
    @brief Calculates the number of possible connections given the quantity of rooms but removes combinations that are garantee to overlap
    @param[in] k number of the total of elements
    @param[in] n number of the total of elements to select
    @return (int) number of combination
*/
int Calculator::NConnectionsReduced(int n){
    int res = 1;
    for(; n > 1; n--)
        res *= 12;

    return res;
}


/*!
    @brief Calculates the number of possible combination of n between k
    @param[in] k number of the total of elements
    @param[in] n number of the total of elements to select
    @return (int) number of combination
*/
int Calculator::NCombination(int k, int n){
    int res = 1;
    int diff = k - n;

    for(; k > n; k--)
        res *= k;

    for(; diff > 1; diff--)
        res /= diff;    

    return res;

}


/*!
    @brief Calculates the number of possible room's size combinations
    @param[in] rooms vector with each room information
    @return (int) number of room's size combinations
*/
int Calculator::NRoomSizes(const std::vector<RoomConfig>& rooms){
    int res = 1;
    const int lenght = rooms.size();
    for(int i = 0; i < lenght; i++){
        const int diffH = rooms[i].maxH - rooms[i].minH;
        const int diffW = rooms[i].maxW - rooms[i].minW;
        const int resH = ((diffH + rooms[i].step  + rooms[i].step - 1) / rooms[i].step);
        const int resW = ((diffW + rooms[i].step  + rooms[i].step - 1) / rooms[i].step);

        res *= resH * resW;
    }

    return res;
}


/*!
    @brief Calculates the size of the upper (or lower) half of a matrix of size n
    @param[in] n size of matrix
    @return (int) lenght of upper half of matrix
*/
int Calculator::lenghtHalfMatrix(const int n){
    return ((n*n) + n) / 2;
}


/*!
    @brief Calculates the total number of possible layouts considering combination, rooms sizes, etc
    @param[in] setups every room configuration
    @param[in] n number of rooms per layout
    @return (int) number of possible layouts
*/
void Calculator::totalOfCombinations(const std::vector<RoomConfig>& setups, const int n){
    std::vector<std::vector<RoomConfig>> allCombs = Iter::getAllComb(setups, n);  

    unsigned long res = 0;
    unsigned long resReduced = 0;
    const unsigned long NPerms = (unsigned long)Calculator::Factorial(n);
    const unsigned long NConns = (unsigned long)Calculator::NConnections(n);
    const unsigned long NConnsReduced = (unsigned long)Calculator::NConnectionsReduced(n);
    const unsigned long qtdPerSize = NPerms * NConns;
    const unsigned long qtdPerSizeReduced = NPerms * NConnsReduced;
    std::cout << "N: " << n << std::endl;
    std::cout << "NPerms: " << NPerms << std::endl;
    std::cout << "NConns: " << NConns << std::endl;
    std::cout << "NConnsReduced: " << NConnsReduced<< std::endl;
  

    for(std::size_t i = 0; i < allCombs.size(); i++){
        res += (unsigned long)(Calculator::NRoomSizes(allCombs[i])) * qtdPerSize;
        resReduced += (unsigned long)(Calculator::NRoomSizes(allCombs[i])) * qtdPerSizeReduced;
    }
    std::cout << "avg NSizes: " << res/(qtdPerSize * allCombs.size()) << std::endl;
    std::cout << "NCombs: " << allCombs.size() << std::endl;
    std::cout << "qtdPerComb: " << res/allCombs.size() << std::endl;
    std::cout << "totalOfCombinations: " << res << std::endl;
    std::cout << "totalOfCombinationsReduced: " << resReduced << std::endl;
}


// void Calculator::listOfCombinations(const std::vector<RoomConfig>& setups, const std::vector<int>& k){
//     std::vector<std::vector<RoomConfig>> allCombs = Iter::getAllComb(setups, n);  

//     unsigned long res = 0;
//     unsigned long resReduced = 0;
//     const unsigned long NPerms = (unsigned long)Calculator::Factorial(n);
//     const unsigned long NConns = (unsigned long)Calculator::NConnections(n);
//     const unsigned long NConnsReduced = (unsigned long)Calculator::NConnectionsReduced(n);
//     const unsigned long qtdPerSize = NPerms * NConns;
//     const unsigned long qtdPerSizeReduced = NPerms * NConnsReduced;
//     std::cout << "N: " << n << std::endl;
//     std::cout << "NPerms: " << NPerms << std::endl;
//     std::cout << "NConns: " << NConns << std::endl;
//     std::cout << "NConnsReduced: " << NConnsReduced<< std::endl;
  

//     for(std::size_t i = 0; i < allCombs.size(); i++){
//         res += (unsigned long)(Calculator::NRoomSizes(allCombs[i])) * qtdPerSize;
//         resReduced += (unsigned long)(Calculator::NRoomSizes(allCombs[i])) * qtdPerSizeReduced;
//     }
//     std::cout << "avg NSizes: " << res/(qtdPerSize * allCombs.size()) << std::endl;
//     std::cout << "NCombs: " << allCombs.size() << std::endl;
//     std::cout << "qtdPerComb: " << res/allCombs.size() << std::endl;
//     std::cout << "totalOfCombinations: " << res << std::endl;
//     std::cout << "totalOfCombinationsReduced: " << resReduced << std::endl;
// }