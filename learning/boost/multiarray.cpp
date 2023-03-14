#include <iostream>
#include <time.h>
#include "boost/array.hpp"
#include "boost/multi_array.hpp"

using namespace boost;

enum {N=1024};

typedef multi_array<char,3> M;
typedef array<array<array<char,N>,N>,N> C;

// Forward declare to avoid being optimised away
static void clear(M& m);
static void clear(C& c);

int main(int,char**)
{
  const clock_t t0=clock();

  {
    M m(extents[N][N][N]);
    clear(m);
  }

  const clock_t t1=clock();

  {
    std::unique_ptr<C> c(new C);
    clear(*c);
  }

  const clock_t t2=clock();

  std::cout 
    << "multi_array: " << (t1-t0)/static_cast<float>(CLOCKS_PER_SEC) << "s\n"
    << "array      : " << (t2-t1)/static_cast<float>(CLOCKS_PER_SEC) << "s\n";

  return 0;
}

void clear(M& m)
{
  for (M::index i=0;i<N;i++)
    for (M::index j=0;j<N;j++)
      for (M::index k=0;k<N;k++)
    m[i][j][k]=1;
}


void clear(C& c)
{
  for (int i=0;i<N;i++)
    for (int j=0;j<N;j++)
      for (int k=0;k<N;k++)
    c[i][j][k]=1;
}