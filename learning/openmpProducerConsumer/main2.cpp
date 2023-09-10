#include <iostream>
#include <vector>
#include <omp.h>

int busy;
// Function to simulate consuming an item
void consumeItem(int item) {
    int threadId = omp_get_thread_num();
    if(threadId == 0){
        printf("%d - Consumed init %d\n", threadId, item);
    } 
    else if (threadId == 1){
        printf("                          %d - Consumed init %d\n", threadId, item);
    }
    else if (threadId == 2){
        printf("                                                    %d - Consumed init %d\n", threadId, item);

    }
    else if (threadId == 3){
        printf("                                                                              %d - Consumed init %d\n", threadId, item);

    }


    // printf("%d - Consumed init %d\n", threadId, item);
    for(int j = 0; j < 500000000 * (threadId + 1); j++);
    // std::cout << "Consumed: " << item << std::endl;
    // printf("%d - Consumed end %d\n", threadId, item);

    if(threadId == 0){
        printf("%d - Consumed end %d\n", threadId, item);
    } 
    else if (threadId == 1){
        printf("                          %d - Consumed end %d\n", threadId, item);
    }
    else if (threadId == 2){
        printf("                                                    %d - Consumed end %d\n", threadId, item);

    }
    else if (threadId == 3){
        printf("                                                                              %d - Consumed end %d\n", threadId, item);

    }

    #pragma omp atomic
    busy--;
}

void produceItem(int item){
    int threadId = omp_get_thread_num();

    if(threadId == 0){
        printf("%d - produced init %d\n", threadId, item);
    } 
    else if (threadId == 1){
        printf("                          %d - produced init %d\n", threadId, item);
    }
    else if (threadId == 2){
        printf("                                                    %d - produced init %d\n", threadId, item);

    }
    else if (threadId == 3){
        printf("                                                                              %d - produced init %d\n", threadId, item);
    }

    for(int j = 0; j < 5000000; j++);

    if(threadId == 0){
        printf("%d - produced %d\n", threadId, item);
    } 
    else if (threadId == 1){
        printf("                          %d - produced %d\n", threadId, item);
    }
    else if (threadId == 2){
        printf("                                                    %d - produced %d\n", threadId, item);

    }
    else if (threadId == 3){
        printf("                                                                              %d - produced %d\n", threadId, item);
    }


    #pragma omp task priority(10)
    {
        consumeItem(item);
    }

}

int main() {
    int itemsToProduce = 5000;
    int dependencyControl = 0;
    // uint8_t* dependencyControl;

    // dependencyControl = (uint8_t*)calloc(itemsToProduce, 0);

    // Initialize OpenMP parallel region
    #pragma omp parallel num_threads(4)
    {

        #pragma omp single
        {
            #pragma omp task depend(inout: dependencyControl) priority(0)
            {
                produceItem(0);
            }

            for(int i = 1; i < itemsToProduce; i++){
                #pragma omp task depend(inout: dependencyControl) priority(0)
                {
                    produceItem(i);
                }
            }
        }
    }

    return 0;
}