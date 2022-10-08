

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cctype>

#include "oneapi/tbb/parallel_pipeline.h"
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/tick_count.h"
#include "oneapi/tbb/tbb_allocator.h"
#include "oneapi/tbb/global_control.h"



int data = 0;

int* generateData(){
    int worker_index = oneapi::tbb::detail::d1::current_thread_index();
    printf("generateData %d\n", worker_index);
    
    int *result = (int*)calloc(40000000, sizeof(int));
    for(int i = 0; i < 4; i++){
        result[i] = worker_index * (i + 1);
    }

    for(int i = 0; i < 100000000 * (worker_index + 1); i++);
    // std::this_thread::sleep_for(1000ms);

    return result;
}

void saveData(int* data){
    int worker_index = oneapi::tbb::detail::d1::current_thread_index();
    if(data == nullptr){
        printf("saveData %d null\n", worker_index);
    } else {
        printf("saveData %d (%d, %d, %d, %d)\n", worker_index, data[0], data[1], data[2], data[3]);
    }

    free(data);

    for(int i = 0; i < 1000000; i++);
}

int main() {
    oneapi::tbb::parallel_pipeline(/*specify max number of bodies executed in parallel, e.g.*/16,
        oneapi::tbb::make_filter<void, int*>(
            oneapi::tbb::filter_mode::parallel, // read data sequentially
            [](oneapi::tbb::flow_control& fc) -> int* {
                // if ( data > 10) {
                //     fc.stop();
                //     return int(); // return dummy value
                // }
                int* input_data = generateData();
                // if(input_data[0] == 3){
                //     fc.stop();
                //     return nullptr; // return dummy value
                // }
                return input_data;
            }
        ) &
        oneapi::tbb::make_filter<int*, void>(
            oneapi::tbb::filter_mode::parallel, // process data in parallel by the first manipulator
            [](int* elem)  {
               saveData(elem);
            }
        ) 
    );
    return 0;
}