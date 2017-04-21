#ifndef TRACKER_TRAINER_NEW_H
#define TRACKER_TRAINER_NEW_H


#include <vector>
#include <opencv/cv.h>

#include "helper/bounding_box.h"
#include "tracker/tracker.h"
#include "network/regressor_train_base.h"
#include "helper/Common.h"

class TrackerTrainerMultiDomain
{
public:
  TrackerTrainerMultiDomain(ExampleGenerator* example_generator);

  TrackerTrainerMultiDomain(ExampleGenerator* example_generator,
                 RegressorTrainBase* regressor_train);

  // Train from this example.
  // Inputs: previous image, current image, previous image's bounding box, current image's bounding box.
  void Train(const cv::Mat& image_prev, const cv::Mat& image_curr,
             const BoundingBox& bbox_prev, const BoundingBox& bbox_curr);

  // Number of total batches trained on so far.
  int get_num_batches() { return num_batches_; }

  // Set current k
  void set_current_k(int k) { current_k_ = k; }

  // get current k 
  int get_current_k() { return current_k_; }

  // get if full batch
  bool get_if_full_batch();

  // set batch_filled_
  void set_batch_filled(bool val);

  // clear remaining data in the batch
  void clear_batch_remaining();

private:
  // Generate training examples and return them.
  // Note that we do not clear the input variables, so if they already contain
  // some examples then we will append to them.
  virtual void MakeTrainingExamples(std::vector<cv::Mat>* images,
                                          std::vector<cv::Mat>* targets,
                                          std::vector<BoundingBox>* bboxes_gt_scaled,
                                          std::vector<std::vector<BoundingBox> > *candidates, 
                                          std::vector<std::vector<double> >  *labels);

  // Train on the batch.
  virtual void ProcessBatch();

  // Data in the current training batch.
  std::vector<cv::Mat> image_currs_batch_;
  std::vector<cv::Mat> images_batch_; // each images_batch_[i] will be repeated according to labels_batch_[i].size()
  std::vector<cv::Mat> targets_batch_; // each targets_batch_[i] will be repeated according to labels_batch_[i].size()
  std::vector<BoundingBox> bboxes_gt_scaled_batch_;
  std::vector<std::vector<double> > labels_batch_;
  std::vector<std::vector<BoundingBox> > candidates_batch_;

  // Used to generate additional training examples through synthetic transformations.
  ExampleGenerator* example_generator_;

  // Neural network.
  RegressorTrainBase* regressor_train_;

  // Number of total batches trained on so far.
  int num_batches_;

  // current domain
  int current_k_;

  // boolean indicating if batch as been filled 
  bool batch_filled_; 
};

#endif // TRACKER_TRAINER_H
