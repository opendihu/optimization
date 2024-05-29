
#include "gtest/gtest.h"

#if __cplusplus >= 201103L && defined __PGI
__attribute__((weak))
void operator delete(void * ptr, unsigned long){ ::operator delete(ptr);}
__attribute__((weak))
void operator delete[](void * ptr, unsigned long){ ::operator delete(ptr);}
#endif  // __cplusplus >= 201103L

namespace {
// Tests that the Foo::Bar() method does Abc.
TEST(CompileTest, GTestCompiles) {
  ASSERT_EQ(true, true);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

