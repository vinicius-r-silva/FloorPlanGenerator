#ifndef GLOBALS
#define GLOBALS

/**
 * Size of the char array for the RoomConfig Name field
 */
#define ROOM_NAME_SIZE 30

/**
 * Enbale/Disable opencv functions
 * necessary to simplify valgrind output
 */
#define OPENCV_ENABLED

#define MULTI_THREAD

// #define N_ROOMS 3

/**
 * @brief Room Config struct containing all information necessary for describing a room setup
 */
typedef struct{
    /*@{*/
    long id;                        /**< Unique Id. Every Id is a factor of two (1, 2, 4, 8, ....)  */
    long depend;                    /**< Reference to another setup unique id, where this current setup only exist if the another one also exists */
    int step;                       /**< int step for iterate to every possible room size */
    int minH;                       /**< minimum Height */
    int maxH;                       /**< maximum Height */
    int minW;                       /**< minimum Width */
    int maxW;                       /**< maximum Width */
    int minRepetitions;             /**< minimum number of repetitions */
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