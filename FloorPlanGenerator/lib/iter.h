#ifndef COMBINATION
#define COMBINATION

#include <globals.h>
#include <vector>

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
};

#endif //COMBINATION