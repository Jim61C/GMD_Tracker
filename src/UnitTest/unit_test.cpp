// shared_ptr::reset example
#include <iostream>
#include <memory>
#include <boost/shared_ptr.hpp>
#include <helper/bounding_box.h>
using namespace std;

int main (int argc, char *argv[]) {
  boost::shared_ptr<BoundingBox> sp;  // empty

  sp.reset (new BoundingBox(1,1,1,1));       // takes ownership of pointer
  std::cout << sp->x1_ << '\n';
  std::cout << sp->y1_ << '\n';
  std::cout << sp->x2_ << '\n';
  std::cout << sp->y2_ << '\n';

  boost::shared_ptr<BoundingBox> sp_2;
  sp_2 = sp;

  cout << "before reset, sp_2:" << endl;
  std::cout << sp_2->x1_ << '\n';
  std::cout << sp_2->y1_ << '\n';
  std::cout << sp_2->x2_ << '\n';
  std::cout << sp_2->y2_ << '\n';

  sp.reset (new BoundingBox(3,3,3,3));       // deletes managed object, acquires new pointer
  std::cout << sp->x1_ << '\n';
  std::cout << sp->y1_ << '\n';
  std::cout << sp->x2_ << '\n';
  std::cout << sp->y2_ << '\n';

  cout << "after reset sp, sp_2:" << endl;
  std::cout << sp_2->x1_ << '\n';
  std::cout << sp_2->y1_ << '\n';
  std::cout << sp_2->x2_ << '\n';
  std::cout << sp_2->y2_ << '\n';

//   sp.reset();               // deletes managed object

  return 0;
}