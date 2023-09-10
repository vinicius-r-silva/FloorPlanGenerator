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

int main() {
    busy = 0;

    // Initialize OpenMP parallel region
    #pragma omp parallel num_threads(4)
    {

        #pragma omp single
        {
            for(int i = 2; i < 100; i++){
                for(int j = 0; j < 50000000; j++);
                int threadId = omp_get_thread_num();
                #pragma omp atomic
                busy++;

                if(threadId == 0){
                    printf("%d - produced %d, busy: %d\n", threadId, i, busy);
                } 
                else if (threadId == 1){
                    printf("                          %d - produced %d, busy: %d\n", threadId, i, busy);
                }
                else if (threadId == 2){
                    printf("                                                    %d - produced %d, busy: %d\n", threadId, i, busy);

                }
                else if (threadId == 3){
                    printf("                                                                              %d - produced %d, busy: %d\n", threadId, i, busy);

                }



                #pragma omp critical
                {
                    if(busy == 4){
                        consumeItem(i);
                    } else {
                        #pragma omp task    
                        consumeItem(i);
                    }
                }
            }
        }
    }

    return 0;
}