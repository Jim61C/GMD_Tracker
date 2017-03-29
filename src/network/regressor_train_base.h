#ifndef REGRESSOR_TRAIN_BASE_H
#define REGRESSOR_TRAIN_BASE_H

#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <caffe/caffe.hpp>
#include <caffe/sgd_solvers.hpp>

#include "helper/bounding_box.h"
#include "network/regressor_base.h"

// We subclass the Caffe solver object so that we can set protected variables like net_ and test_nets_.
// (Admittedly, this is probably not the intended use of the abstraction).
class MySolver : public caffe::SGDSolver<float>
{
public:
  MySolver(const std::string& param_file);

  void set_net(const boost::shared_ptr<caffe::Net<float> >& net) {
    net_ = net;
  }
  void set_test_net(const boost::shared_ptr<caffe::Net<float> >& net) {
    test_nets_[0] = net;
  }
};

// The class used to train the tracker should inherit from this class.
class RegressorTrainBase 
{
public:
  RegressorTrainBase(const std::string& solver_file);

  // Train the tracker.
  virtual void Train(const std::vector<cv::Mat>& images,
             const std::vector<cv::Mat>& targets,
             const std::vector<BoundingBox>& bboxes_gt) = 0;
  
  // Train the tracker, GOTURN, MDNet, k = -1 indicating Fine Tuning
  virtual void TrainBatch(const std::vector<cv::Mat>& images,
                           const std::vector<cv::Mat>& targets,
                           const std::vector<BoundingBox>& bboxes_gt,
                           const std::vector<std::vector<cv::Mat> > &candidates,
                           const std::vector<std::vector<double> > &labels,
                           int k) = 0;

  // TODO: add an interface for fine-tuning, just one domain and no bboxes_gt (or dummy bboxes_gt);
  
  // Interface for saving the loss_history, implementation depends on implementing sub-classes
  virtual void SaveLossHistoryToFile(const std::string &save_path) = 0;

protected:
  MySolver solver_;
};

#endif // REGRESSOR_TRAIN_BASE_H
