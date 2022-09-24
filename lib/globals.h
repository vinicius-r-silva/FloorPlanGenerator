#ifndef GLOBALS
#define GLOBALS

#define ROOM_NAME_SIZE 30
#define N_ROOMS 3

typedef struct{
    long id;
    int step;
    int numExtensions;
    int minH; int maxH; 
    int minW; int maxW;
    char name[ROOM_NAME_SIZE];
} RoomConfig;

// RoomConfig *

#endif //GLOBALS