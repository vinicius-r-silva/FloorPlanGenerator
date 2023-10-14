#ifndef VIEWER
#define VIEWER
#include <vector>
#include <string>

/** 
 * @brief Viewer existing connections
*/
class Viewer
{

public:
    /** 
     * @brief Viewer Constructor
     * @return None
    */
    Viewer();

    static void showCoreResults(const std::vector<int16_t>& arr, const int n);
    
    static void showFileResults(std::string fullPath, int arrayOffset, int ptsPerLayout);

    static void showResults(std::string fullPath);
};

#endif //VIEWER