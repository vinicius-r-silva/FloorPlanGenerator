// http://www.inf.ufsc.br/~bosco.sobral/ensino/ine5645/Conceitos_OpenMP.pdf
// https://www.ibm.com/docs/en/zos/2.2.0?topic=programs-shared-private-variables-in-parallel-environment

#include<iostream>
#include <vector>
#include <algorithm>   
#include <omp.h>
#include <sstream>    

// https://stackoverflow.com/questions/11071116/i-got-omp-get-num-threads-always-return-1-in-gcc-works-in-icc
int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
        n += 1;
    
    return n;
}

int main( int ac, char **av){
    std::vector<int> permutation_base{1, 2, 3, 4};
    int n = permutation_base.size();

    int N1 = omp_thread_count();
    std::cout << "omp_thread_count " << N1 << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) 
    {
        // Make a copy of permutation_base
        auto perm = permutation_base;
        std::stringstream ss;
        int tid = omp_get_thread_num();
        // rotate the i'th  element to the front
        // keep the other elements sorted
        
        std::rotate(perm.begin(), perm.begin() + i, perm.begin() + i + 1);
        // Now go through all permutations of the last `n-1` elements. 
        // Keep the first element fixed. 
        do {
            ss << "Thread: " << tid << ", perm: ";
            for (int j: perm)
                ss << j << ' ';
            ss << std::endl;
        }
        while (std::next_permutation(perm.begin() + 1, perm.end()));
        std::cout << ss.str();
    }
}