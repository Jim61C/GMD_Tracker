#ifndef REGRESSOR_TRAIN_H
#define REGRESSOR_TRAIN_H

#include "network/regressor.h"
#include "network/regressor_train_base.h"
#include "helper/CommonCV.h"
#include "helper/helper.h"

class RegressorTrain : public Regressor, public RegressorTrainBase
{
public:
  RegressorTrain(const std::string& deploy_proto,
                 const std::string& caffe_model,
                 const int gpu_id,
                 const std::string& solver_file,
                 const int num_input,
                 const bool do_train);     

  RegressorTrain(const std::string& deploy_proto,
                 const std::string& caffe_model,
                 const int gpu_id,
                 const std::string& solver_file,
                 const bool do_train);            

  RegressorTrain(const std::string& deploy_proto,
                 const std::string& caffe_model,
                 const int gpu_id,
                 const std::string& solver_file);

  RegressorTrain(const std::string& deploy_proto,
                               const std::string& caffe_model,
                               const int gpu_id,
                               const string& solver_file,
                               const string& loss_save_path,
                               const int K);

  // Train the tracker.
  void Train(const std::vector<cv::Mat>& images,
                             const std::vector<cv::Mat>& targets,
                             const std::vector<BoundingBox>& bboxes_gt);
  // Actuall setup on batch
  void Train(std::vector<cv::Mat> &images_flattened,
                           std::vector<cv::Mat> &targets_flattened,
                           std::vector<cv::Mat> &candidates_flattened,
                           std::vector<double> &labels_flattened,
                           int k);
  
  void TrainForwardBackwardWorker(const cv::Mat & image_curr,
                          const std::vector<BoundingBox> &candidates_bboxes, 
                          const std::vector<double> &labels,
                          const cv::Mat & image,
                          const cv::Mat & target,
                          int k,
                          int num_nohem);

  void TrainForwardBackward(const cv::Mat & image_curr,
                          const std::vector<BoundingBox> &candidates_bboxes, 
                          const std::vector<double> &labels_flattened,
                          const cv::Mat & image,
                          const cv::Mat & target,
                          int k,
                          int num_nohem);
  
  // Forward and Backward. TODO: add Online Hard Example Mining
  void TrainBatchFast(const std::vector<cv::Mat>& image_currs,
                           const std::vector<cv::Mat>& images,
                           const std::vector<cv::Mat>& targets,
                           const std::vector<BoundingBox>& bboxes_gt,
                           const std::vector<std::vector<BoundingBox> > &candidate_bboxes,
                           const std::vector<std::vector<double> > &labels,
                           int k,
                           int inner_batch_size = INNER_BATCH_SIZE,
                           int num_nohem = -1);

  // Implementing the TrainBatch Interface
  void TrainBatch(const std::vector<cv::Mat>& images,
                           const std::vector<cv::Mat>& targets,
                           const std::vector<BoundingBox>& bboxes_gt,
                           const std::vector<std::vector<cv::Mat> > &candidates,
                           const std::vector<std::vector<double> > &labels,
                           int k);
  
  // Implementing the loss saving Interface                  
  void SaveLossHistoryToFile(const std::string &save_path);

  void InvokeSaveLossIfNeeded();

  // Set up the solver with the given test file for validation testing.
  void set_test_net(const std::string& test_proto);

  // Set the labels in the net_'s input[3]
  void set_labels(const vector<double>  &labels_flattened);

  // Reset the solver's net to this->net_ initialised from regressor 
  void ResetSolverNet();

private:
  // Train the network.
  void Step();

  // Set the ground-truth bounding boxes (for training).
  void set_bboxes_gt(const std::vector<BoundingBox>& bboxes_gt);

  boost::shared_ptr<caffe::Net<float> > test_net_;

  std::vector<double> loss_history_;

  const std::string loss_save_path_;
};

#endif // REGRESSOR_TRAIN_H
