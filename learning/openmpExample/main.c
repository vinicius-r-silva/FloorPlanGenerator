// http://www.inf.ufsc.br/~bosco.sobral/ensino/ine5645/Conceitos_OpenMP.pdf
// https://www.ibm.com/docs/en/zos/2.2.0?topic=programs-shared-private-variables-in-parallel-environment

#include<stdio.h>

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
     
    #pragma omp parallel num_threads(N1)
    { 
        int tid = omp_get_thread_num();
        printf("Thread %d: Hello!\n", tid);
    }
    return 0;
}