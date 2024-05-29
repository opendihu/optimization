

#if __cplusplus >= 201103L && defined __PGI
__attribute__((weak))
void operator delete(void * ptr, unsigned long){ ::operator delete(ptr);}
__attribute__((weak))
void operator delete[](void * ptr, unsigned long){ ::operator delete(ptr);}
#endif  // __cplusplus >= 201103L

#include <iostream>
using namespace std;

// Include namespace SEMT & global operators.
#include "semt/Semt.h"
// Include macros: INT, DINT, VAR, DVAR, PARAM, DPARAM
#include "semt/Shortcuts.h"
using namespace SEMT;

template<typename T>
std::ostream &operator<<(std::ostream &stream, const std::vector<T> &values)
{
  if (values.empty())
  {
    stream << "()";
    return stream;
  }

  stream << "(" << values[0];
  for (unsigned long i=1; i<values.size(); i++)
    stream << "," << values[i];
  stream << ")";
  return stream;
}

int main(int argc, char* argv[])
{
  DVAR(x2, 2);
  DINT(Two, 2);
  DPARAM(t1, 1)
  cout << (VAR(0) * VAR(1) - t1 + pow(x2, Two)) << endl;
  cout << deriv_t(pow(VAR(0) * x2, PARAM(0)), x2) << endl;
// output:
// (((x0 * x1) - t1) + (x2)^(2))
// (((x0 * x2))^(t0) * ((t0 * x0) / (x0 * x2)))
  return EXIT_SUCCESS;
}

