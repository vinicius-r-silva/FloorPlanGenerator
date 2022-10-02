// http://www.inf.ufsc.br/~bosco.sobral/ensino/ine5645/Conceitos_OpenMP.pdf
// https://www.ibm.com/docs/en/zos/2.2.0?topic=programs-shared-private-variables-in-parallel-environment

#include<stdlib.h>
#include<stdio.h>
#include <omp.h>

// https://stackoverflow.com/questions/11071116/i-got-omp-get-num-threads-always-return-1-in-gcc-works-in-icc
int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
        n += 1;
    
    return n;
}


int main( int ac, char **av){
    int N1 = omp_thread_count();
    printf("omp_thread_count %d\n", N1);

    int i = 0;
    const int arraySize = N1 * 4;
    int *array = (int*)calloc(arraySize,  sizeof(int));
    printf("arraySize %d\n", arraySize);
     
    #pragma omp parallel for num_threads(N1) default(none) shared(array, arraySize)
        for(i = 0; i < arraySize; i++)
    { 
        const int tid = omp_get_thread_num();
        printf("Thread %d: i = %d\n", tid, i);
    }
    return 0;
}