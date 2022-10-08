// http://www.inf.ufsc.br/~bosco.sobral/ensino/ine5645/Conceitos_OpenMP.pdf
// https://www.ibm.com/docs/en/zos/2.2.0?topic=programs-shared-private-variables-in-parallel-environment
// g++ main.c -o main -fopenmp -lpthread

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <thread>
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;


// https://stackoverflow.com/questions/11071116/i-got-omp-get-num-threads-always-return-1-in-gcc-works-in-icc
int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
        n += 1;
    
    return n;
}

void test0();
void test1();
void test2();
void test3();
void test4();



int main( int ac, char **av){
    test1();
    // #pragma omp parallel for num_threads(N1) default(none) shared(array, arraySize)
    //     for(i = 0; i < arraySize; i++)
    // { 
    //     const int tid = omp_get_thread_num();
    //     printf("Thread %d: i = %d\n", tid, i);
    //     std::this_thread::sleep_for(1000ms);
    // }
    return 0;
}

void test1(){
    int x = 0;
    int *array = (int*)calloc(16, sizeof(int));
    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            int a = 0, b = 0, c = 0, d = 0, e = 0;
            for(int j = 0; j < 12; j++){
                #pragma omp task depend(out: array[j]) priority(10)
                {
                    const int tid = omp_get_thread_num();
                    printf("A %d init (%d)\n", tid, j);
                    for(int k = 0; k < 100000000; k++);
                    array[j] = j*2;
                    // std::this_thread::sleep_for(1000ms);

                    // x += 1;
                    // printf("A is finished x: %d\n", x);
                }
                printf("new task %d\n", j);
                #pragma omp task depend(in: array[j]) priority(20)
                {
                    const int tid = omp_get_thread_num();
                    printf("E %d init (%d) - %d\n", tid, j, array[j]);
                    // printf("E is finished\n");
                }
            }

        }
    }
}

void test0(){

    int N1 = omp_thread_count();
    printf("omp_thread_count %d\n", N1);

    int i = 0;
    int x = 0;
    const int arraySize = N1 * 4;
    int **array = (int**)calloc(arraySize,  sizeof(int*));
    printf("arraySize %d\n", arraySize);
     
    #pragma omp parallel num_threads(3)
    {
        #pragma omp single nowait
        {
            for (int64_t i=0; i<arraySize; i++) {
                #pragma omp task priority(10) depend(out: array[i])
                {
                    const int tid = omp_get_thread_num();
                    printf("%d init (%ld)\n", tid, i);
                    array[i] = (int*)calloc(1,  sizeof(int));
                    *(array[i]) = i * 2;
                    std::this_thread::sleep_for(1000ms);
                    // (void) read_input(&status_read[i],….);
                }
                #pragma omp task priority(20) depend(in: array[i])
                {
                    const int tid = omp_get_thread_num();
                    int value = *(array[i]);
                    printf("%d  -  %ld\n", tid, i);
                    std::this_thread::sleep_for(100ms);
                    // std::this_thread::sleep_for(1000ms);
                    // printf("%d end\n", tid);
                    // (void) compute_results(…);
                }
            } // End of for-loop
        } // End of single region
    } // End of parallel region
}