#ifndef GLOBALS
#define GLOBALS

#define ROOM_NAME_SIZE 30

typedef struct{
    long id;
    int numExtensions;
    int minH; int maxH; 
    int minW; int maxW;
    char name[ROOM_NAME_SIZE];
} RoomConfig;

// RoomConfig *

#endif //GLOBALS