// shared_ptr::reset example
#include <iostream>
#include <memory>
#include <boost/shared_ptr.hpp>
#include <helper/bounding_box.h>
#include <helper/CommonCV.h>
using namespace std;

class A {
public:
  virtual void foo() {
    cout << "A's foo" << endl;
  }
  virtual void foo_invoker() {
    cout << "A's foo_invoker, about to call foo()" << endl;
    foo();
  }
};

class B: public A {
public:
  virtual void foo() {
    cout << "B's foo" << endl;
  }
  virtual void foo_invoker() {
    A::foo_invoker();
  }
};

void doSomethingMat(Mat & mat) {
  Mat M = (Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  M.copyTo(mat);
}

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

  vector<int> temp = {1,2,3,4,5,6,7,8,9,10};
  // auto engine = std::default_random_engine{};
  cout << "time(NULL) " << time(NULL) << endl;
  // std::mt19937 engine(time(NULL));
  std::mt19937 engine;
  engine.seed(time(NULL));

  for (int i = 0; i < 5 ; i ++) {
    vector<int> this_temp(temp);
    std::shuffle(this_temp.begin(), this_temp.end(), engine);
    for (int j = 0; j < this_temp.size() ; j ++) {
      cout << this_temp[j] << " ";
    }
    cout << endl;
  }


  cout << std::min(1,2) << endl;

  // TODO: unit test assign &Bbox to a bbox, creates a copy!
  BoundingBox box_a(0, 0, 255, 255);
  BoundingBox &box_b = box_a;

  BoundingBox box_c;
  box_c = box_b;

  box_a.x1_ = 10;
  box_a.y1_ = 10;

  cout << "box_a:" << box_a.x1_ << ", "<< box_a.y1_ << ", " << box_a.x2_ << ", " << box_a.y2_ << endl;
  cout << "box_b:" << box_b.x1_ << ", "<< box_b.y1_ << ", " << box_b.x2_ << ", " << box_b.y2_ << endl;
  cout << "box_c:" << box_c.x1_ << ", "<< box_c.y1_ << ", " << box_c.x2_ << ", " << box_c.y2_ << endl; 


  // TODO: unit test dynamic binding for tracker_->Init, which function to call will be based on the dynamic type, virtual table looked up during runtime
  A a;
  B b;
  cout << "a.foo_invoker():" << endl;
  a.foo_invoker();
  cout << "b.foo_invoker():" << endl;
  b.foo_invoker();

  // Test memcpy
  vector<float> src = {1, 2, 3, 4};
  vector<float> dst (src);
  // memcpy(&dst, &src, 4);
  cout << "dst.size(): " << dst.size() << endl;
  
  for (int i =0 ; i < 4; i ++) { // note that beyond will be undefined
    cout << dst[i] << " ";
  }
  cout << endl;

  Mat temp_mat = (Mat_<double>(3,3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);
  doSomethingMat(temp_mat);
  cout << "after doSomethingMat: \n" << temp_mat << endl;
  temp_mat.at<double>(2,2) = 10;
  cout << "after assign (2, 2): \n" << temp_mat << endl;

  doSomethingMat(temp_mat);
  cout << "after doSomethingMat: \n" << temp_mat << endl;

  return 0;
}