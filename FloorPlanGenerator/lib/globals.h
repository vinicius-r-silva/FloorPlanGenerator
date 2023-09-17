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
    int familyIds;
    int rPlannyId;                  /**< rPlanny Id */
    // int nameId;                     /**< name Id */
    // int minRepetitions;             /**< minimum number of repetitions */
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

// #define REQ_TYPE 13
// #define REQ_AND 0
// #define REQ_OR  1

// enum roomNameIds {
//     _ID_SALA = 0,
//     _ID_BANHEIRO = 1,
//     _ID_COZINHA = 2,
//     _ID_LAVANDERIA = 3,
//     _ID_QUARTO = 4,
//     _ID_QUARTO_2 = 5,
//     _NAME_IDS_COUNT = 6
// };



#define __GENERATE_N 3
#define __GENERATE_PERM 6 // __GENERATE_N!
#define __GENERATE_REQ_ADJ 9 // __GENERATE_N * __GENERATE_N
#define __GENERATE_ROTATIONS 8 //2 ^ __GENERATE_N
#define __GENERATE_RES_LENGHT 13 // __GENERATE_N * 4 + 1
#define __GENERATE_SIZE_LENGHT 6 //__GENERATE_N * 2
#define __GENERATE_RES_LAYOUT_LENGHT 12 // __GENERATE_N * 4

#define __ROOM_CONFIG_STEP 0
#define __ROOM_CONFIG_MINH 1
#define __ROOM_CONFIG_MAXH 2
#define __ROOM_CONFIG_MINW 3
#define __ROOM_CONFIG_MAXW 4
#define __ROOM_CONFIG_COUNTH 5
#define __ROOM_CONFIG_COUNTW 6
#define __ROOM_CONFIG_RID 7

#define __ROOM_CONFIG_LENGHT 8


#define __COMBINE_N_A 3
#define __COMBINE_N_B 3
#define __COMBINE_PERM_A 6  // __COMBINE_N_A !
#define __COMBINE_PERM_B 6  // __COMBINE_N_B !

// #define __COMBINE_CONN 108 //check on the google sheets, i dont want to think about the formula for this number now
#define __COMBINE_CONN_SRC_X_SHIFT 0
#define __COMBINE_CONN_SRC_Y_SHIFT 4
#define __COMBINE_CONN_DST_X_SHIFT 8
#define __COMBINE_CONN_DST_Y_SHIFT 12
#define __COMBINE_CONN_BITS 0b0000000000001111
// #define __COMBINE_CONN_SRC_Y_BITS 0b0000000011110000
// #define __COMBINE_CONN_DST_X_BITS 0b0000111100000000
// #define __COMBINE_CONN_DST_Y_BITS 0b1111000000000000


#define __CONN_CHECK 63 //1 << 0 | 1 << 1 | .... | 1 << (_N_A + N_B - 1)
#define __CONN_CHECK_IDX 5 // _N_A + N_B - 1

#define __SIZE_A_LAYOUT 12		// __COMBINE_N_A * 4
#define __SIZE_B_LAYOUT 12		// __COMBINE_N_B * 4
#define __SIZE_A_DISK 13 // __SIZE_B + perm iter value
#define __SIZE_B_DISK 13 // __SIZE_B + perm iter value
// #define __SIZE_RES 4
#define __SIZE_RES 24

// TODO INCREASE RPLANNY IDS TO 5 TYPES INSTEAD OF 4 TYPES

#define __SIZE_ADJ_TYPES 4
#define __SIZE_ADJ 16 // Req Adj types * Req Adj types
// #define __SIZE_PERM_A 18 // __COMBINE_N_A * __N_PERM_A
// #define __SIZE_PERM_B 18 // __COMBINE_N_B * __N_PERM_B

#define __LEFT 0
#define __UP 1
#define __RIGHT 2
#define __DOWN 3

#define __RID_BITS_SIZE 3
#define __RID_BITS 7 // 1 << 0 | 1 << 1 | ... | 1 <<  (__RID_BITS_SIZE - 1)

#define __THREADS_PER_BLOCK 768 // 192, 288, 384, 480, 576, 672, 768, 862, 


#endif //GLOBALS