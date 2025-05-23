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
    

int Calculator::NRotations(int n){
    int res = 1;
    for(int i = 1; i <= n; i++){
        res *= 2;
    }

    return res;
}


/*!
    @brief Calculates the number of possible connections given the quantity of rooms
    @details f(2) = 16 || f(n) = f(n-1) * (n - 1) * 16
    @param[in] n input to calculate the number of connections
    @return (int) number of connections
*/
int Calculator::NConnections(int n){
    if(n <= 1)
        return 0;

    if(n == 2)
        return 16;

    int res = NConnections(n - 1) * (n - 1) * 16;
    return res;
}
    

/*!
    @brief Calculates the number of possible connections given the quantity of rooms but removes combinations that are garantee to overlap
    @param[in] k number of the total of elements
    @param[in] n number of the total of elements to select
    @return (int) number of combination
*/
int Calculator::NConnectionsReduced(int n){
    if(n == 1)
        return 0;

    int res = 1;
    for(int i = 1; i < n; i++){
        res *= i * 12;
    }

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

int Calculator::IntersectionArea(
    const int a_up, const int a_down, const int a_left, const int a_right, 
    const int b_up, const int b_down, const int b_left, const int b_right) 
{
    int top = std::max(a_up, b_up);
    int left = std::max(a_left, b_left);
    int bottom = std::min(a_down, b_down);
    int right = std::min(a_right, b_right);

    if (top < bottom && left < right) {
        return (bottom - top) * (right - left);
    } else {
        return 0;
    }
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


std::pair<int, int> Calculator::getMinimumAcceptableBoundingBoxDimensions(const std::vector<int16_t>& input){
    const int tolerance = __SEARCH_TOLERANCE_AREA_PCT;
    
    int totalArea = 0;
	int minH = 5000, maxH = -5000;
	int minW = 5000, maxW = -5000;
	for(size_t i = 0; i < input.size(); i+=4){
        int left = input[i + __LEFT];
        int up = input[i + __UP];
        int down = input[i + __DOWN];
        int right = input[i + __RIGHT];

        totalArea += (down - up) * (right - left);

        minH = std::min(minH, up);
        maxH = std::max(maxH, down);
        minW = std::min(minW, left);
        maxW = std::max(maxW, right);
	}
    const int diffH = maxH - minH;
    const int diffW = maxW - minW;

    int newArea = 0;
    int reduction = 0;
    int reductionPct = 0;
    while(reductionPct <= tolerance){
        reduction++;
        newArea = 0;
        int newMaxH = maxH - reduction;

        for(size_t i = 0; i < input.size(); i+=4){
            int left = input[i + __LEFT];
            int up = input[i + __UP];
            int down = input[i + __DOWN];
            int right = input[i + __RIGHT];

            if(down > newMaxH)
                down = newMaxH;

            if(up < down)
                newArea += (down - up) * (right - left);
        }

        reductionPct = 100 - ((newArea * 100) / totalArea);
    }
    const int reductionFromBottom = reduction - 1;

    reduction = 0;
    reductionPct = 0;
    while(reductionPct <= tolerance){
        reduction++;
        newArea = 0;
        int newMinH = minH + reduction;

        for(size_t i = 0; i < input.size(); i+=4){
            int left = input[i + __LEFT];
            int up = input[i + __UP];
            int down = input[i + __DOWN];
            int right = input[i + __RIGHT];

            if(up < newMinH)
                up = newMinH;

            if(up < down)
                newArea += (down - up) * (right - left);
        }

        reductionPct = 100 - ((newArea * 100) / totalArea);
    }
    const int reductionFromTop = reduction - 1;

    reduction = 0;
    reductionPct = 0;
    while(reductionPct <= tolerance){
        reduction++;
        newArea = 0;
        int newMaxW = maxW - reduction;

        for(size_t i = 0; i < input.size(); i+=4){
            int left = input[i + __LEFT];
            int up = input[i + __UP];
            int down = input[i + __DOWN];
            int right = input[i + __RIGHT];

            if(right > newMaxW)
                right = newMaxW;

            if(left < right)
                newArea += (down - up) * (right - left);
        }

        reductionPct = 100 - ((newArea * 100) / totalArea);
    }
    const int reductionFromRight = reduction - 1;

    reduction = 0;
    reductionPct = 0;
    while(reductionPct <= tolerance){
        reduction++;
        newArea = 0;
        int newMinW = minW + reduction;

        for(size_t i = 0; i < input.size(); i+=4){
            int left = input[i + __LEFT];
            int up = input[i + __UP];
            int down = input[i + __DOWN];
            int right = input[i + __RIGHT];

            if(left < newMinW)
                left = newMinW;

            if(left < right)
                newArea += (down - up) * (right - left);
        }

        reductionPct = 100 - ((newArea * 100) / totalArea);
    }
    const int reductionFromLeft = reduction - 1;

    const int heightTolerance = diffH - std::max(reductionFromBottom, reductionFromTop);
    const int widthTolerance = diffW - std::max(reductionFromRight, reductionFromLeft);
    
    return std::make_pair(heightTolerance, widthTolerance);
}


std::pair<int, int> Calculator::getMaximumAcceptableBoundingBoxDimensions(const std::vector<int16_t>& input){
	int minH = 5000, maxH = -5000;
	int minW = 5000, maxW = -5000;
	for(size_t i = 0; i < input.size(); i+=4){
        int left = input[i + __LEFT];
        int up = input[i + __UP];
        int down = input[i + __DOWN];
        int right = input[i + __RIGHT];

        minH = std::min(minH, up);
        maxH = std::max(maxH, down);
        minW = std::min(minW, left);
        maxW = std::max(maxW, right);
	}
    const int diffH = maxH - minH;
    const int diffW = maxW - minW;

    return std::make_pair(diffH + 15, diffW + 15);

    // const int tolerance = __SEARCH_TOLERANCE_AREA_PCT;
    
    // int totalArea = 0;
	// int minH = 5000, maxH = -5000;
	// int minW = 5000, maxW = -5000;
	// for(size_t i = 0; i < input.size(); i+=4){
    //     int left = input[i + __LEFT];
    //     int up = input[i + __UP];
    //     int down = input[i + __DOWN];
    //     int right = input[i + __RIGHT];

    //     totalArea += (down - up) * (right - left);

    //     minH = std::min(minH, up);
    //     maxH = std::max(maxH, down);
    //     minW = std::min(minW, left);
    //     maxW = std::max(maxW, right);
	// }
    // const int diffH = maxH - minH;
    // const int diffW = maxW - minW;
    
    // const int sectionPctConstant = 5;
    // const int areaTolerance = (totalArea * tolerance) / 10;
    // const int hSection = (diffH * sectionPctConstant) / 10;
    // const int wSection = (diffW * sectionPctConstant) / 10;

    // const int heightTolerance = (areaTolerance / hSection) / 10;
    // const int widthTolerance = (areaTolerance / wSection) / 10;

    // return std::make_pair(heightTolerance, widthTolerance);
}

std::pair<int, int> Calculator::getCenterOfMass(const std::vector<int16_t>& layout){
    int totalArea = 0;
    int centerH = 0;
    int centerW = 0;

	for(size_t i = 0; i < layout.size(); i+=4){
        int left = layout[i + __LEFT] * 10;
        int up = layout[i + __UP] * 10;
        int down = layout[i + __DOWN] * 10;
        int right = layout[i + __RIGHT] * 10;

        const int roomArea = (down - up) * (right - left);
        const int h = (down + up) / 2;
        const int w = (right + left) / 2;

        centerH += roomArea * h;
        centerW += roomArea * w;

        totalArea += roomArea;
	}

    centerH /= totalArea * 10;
    centerW /= totalArea * 10;
    
    return std::make_pair(centerH, centerW);
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