#ifndef GLOBALS
#define GLOBALS

/**
 * Size of the char array for the RoomConfig Name field
 */
#define ROOM_NAME_SIZE 30
// #define N_ROOMS 3

/**
 * @brief Room Config struct containing all information necessary for describing a room setup
 */
typedef struct{
    /*@{*/
    long id;                        /**< Unique Id. Every Id is a factor of two (1, 2, 4, 8, ....)  */
    int step;                       /**< int step for iterate to every possible room size */
    int minH;                       /**< minimum Height */
    int maxH;                       /**< maximum Height */
    int minW;                       /**< minimum Width */
    int maxW;                       /**< maximum Width */
    int minRepetitions;             /**< minimum number of repetitions */
    int maxRepetitions;             /**< maximum number of repetitions */
    char name[ROOM_NAME_SIZE];      /**< room name */
    
    /**
     * number of "outbound" rectangles attached to the room 
     * (every normal room is rectangular, this "attachments" make
     *  it possible to have more the 4 vertexes in a room setup)
     */
    int numExtensions;
    /*@}*/
} RoomConfig;

#endif //GLOBALS