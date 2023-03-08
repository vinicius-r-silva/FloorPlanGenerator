#ifndef GLOBALS
#define GLOBALS

#include <stdint.h>

/**
 * Size of the char array for the RoomConfig Name field
 */
#define ROOM_NAME_SIZE 15
#define SIZE_H_IDX 0
#define SIZE_W_IDX 1
#define PERM_ID_IDX 2
#define CONN_ID_IDX 3


/**
 * Enbale/Disable opencv functions
 * necessary to simplify valgrind output
 */
#define OPENCV_ENABLED

// #define MULTI_THREAD

// #define N_ROOMS 3

/**
 * @brief Room Config struct containing all information necessary for describing a room setup
 */
typedef struct{
    /*@{*/
    long id;                        /**< Unique Id. Every Id is a factor of two (1, 2, 4, 8, ....)  */
    long depend;                    /**< Reference to another setup unique id, where this current setup only exist if the another one also exists */
    int16_t step;                   /**< int step for iterate to every possible room size */
    int16_t minH;                   /**< minimum Height */
    int16_t maxH;                   /**< maximum Height */
    int16_t minW;                   /**< minimum Width */
    int16_t maxW;                   /**< maximum Width */
    int rPlannyId;                  /**< rPlanny Id */
    // int nameId;                     /**< name Id */
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

#define REQ_NONE 0
#define REQ_ANY 1
#define REQ_ALL 2

// enum roomNameIds {
//     _ID_SALA = 0,
//     _ID_BANHEIRO = 1,
//     _ID_COZINHA = 2,
//     _ID_LAVANDERIA = 3,
//     _ID_QUARTO = 4,
//     _ID_QUARTO_2 = 5,
//     _NAME_IDS_COUNT = 6
// };
#endif //GLOBALS