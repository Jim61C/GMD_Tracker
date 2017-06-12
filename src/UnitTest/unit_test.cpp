// shared_ptr::reset example
#include <iostream>
#include <memory>
#include <boost/shared_ptr.hpp>
#include <helper/bounding_box.h>
#include <helper/CommonCV.h>
#include <Eigen/Dense>
#include "../rapidxml/rapidxml.hpp"
#include "../rapidxml/rapidxml_utils.hpp"
#include <assert.h>
#include <string.h>
using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using Eigen::MatrixXf;
using Eigen::Matrix;

using namespace rapidxml;

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

void populateTestEigenMap() {
  // Test Eigen Map
  vector<vector<float> > src;
  for (int i = 0; i < 4; i ++) {
    vector<float> temp;
    for (int j = 0; j < 5; j ++) {
      temp.push_back(j);
    }
    src.push_back(temp);
  }

  Map<Matrix<float,1,5,Eigen::RowMajor> > mf(src[0].data());
  for (int j = 0; j < 5; j ++) {
    cout << mf(0, j) << " ";
  }
  cout << endl;
}

void populateTestEigenFunctions() {
  MatrixXd test(3, 3);
  test <<  4,-1,2, -1,6,0, 2,0,5;

  cout << test << endl;
  cout << "test.col(1).mean():" << test.col(1).mean() << endl;

  Eigen::LLT<MatrixXd> llt(test); // compute the Cholesky decomposition of A
  MatrixXd L = llt.matrixL();

  cout << L << endl;
  cout << L * L.transpose() << endl;

  Eigen::EigenSolver<MatrixXd>eigen_solver(test, true);

  MatrixXd ones = MatrixXd::Ones(3,3);
  Eigen::EigenSolver<MatrixXd> es(ones);
  cout << "The first eigenvector of the 3x3 matrix of ones is:"
      << endl << es.eigenvectors() << endl;
  cout << es.eigenvalues().cols() << endl;


}


void populateTestRapidXml () {
    rapidxml::file<> xmlFile("/home/jimxing/Downloads/Tracking_Sequences/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0004/ILSVRC2015_val_00069001/000000.xml"); // Default template is char
    rapidxml::xml_document<> doc;
    doc.parse<0>(xmlFile.data());

    cout << "first node name:" << doc.first_node()->name() << endl;

    assert(strcmp(doc.first_node()->name(), "annotation") == 0);
    xml_node<> *annotation_node = doc.first_node();
    for (xml_node<> *child = annotation_node->first_node(); child; child = child->next_sibling()) {
      // do stuff with child
      cout << child->name() << endl;
      if (strcmp(child->name(), "object") == 0) {
        for (xml_node<> *track_obj_node = child->first_node(); track_obj_node; track_obj_node = track_obj_node->next_sibling()) {
          if (strcmp(track_obj_node->name(), "trackid") == 0) {
            cout << "\t" << "trackid: " << track_obj_node->value() << endl;
          }
          else if (strcmp(track_obj_node->name(), "name") == 0) {
            cout << "\t" << "name: " << track_obj_node->value() << endl;
          }
          else if (strcmp(track_obj_node->name(), "occluded") == 0) {
            cout << "\t" << "occluded: " << track_obj_node->value() << endl;
          }
          else if (strcmp(track_obj_node->name(), "generated") == 0) {
            cout << "\t" << "generated: " << track_obj_node->value() << endl;
          }
          else if (strcmp(track_obj_node->name(), "bndbox") == 0) {
            xml_node<> *xmax_node = track_obj_node->first_node();
            xml_node<> *xmin_node = xmax_node->next_sibling();
            xml_node<> *ymax_node = xmin_node->next_sibling();
            xml_node<> *ymin_node = ymax_node->next_sibling();

            cout << "\t\tx1: " << xmin_node->value() << endl
            << "\t\ty1: " << ymin_node->value() << endl 
            << "\t\tx2: " << xmax_node->value() << endl
            << "\t\ty2: " << ymax_node->value() << endl;
          }
        }
      }
    }
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

  // populateTestEigenMap();
  // populateTestEigenFunctions();
  populateTestRapidXml();
  

  return 0;
}