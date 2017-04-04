#ifndef TRACKER_GMD_H
#define TRACKER_GMD_H

#include "tracker.h"

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <gsl/gsl_rng.h> 
#include <gsl/gsl_randist.h> /* GAUSSIAN*/
#include "helper/Constants.h"
#include <limits.h>

class TrackerGMD : public Tracker {

public:
  TrackerGMD(const bool show_tracking);

  // Estimate the location of the target object in the current image.
  virtual void Track(const cv::Mat& image_curr, RegressorBase* regressor,
             BoundingBox* bbox_estimate_uncentered);
  
  // Online fine tune, given the networks and example_generators
  virtual void FineTuneOnline(size_t frame_num, ExampleGenerator* example_generator,
                                RegressorTrainBase* regressor_train, bool success_frame, bool is_last_frame);
  
  // Actual worker to do the finetune
  void FineTuneWorker(ExampleGenerator* example_generator,
                                RegressorTrainBase* regressor_train,
                                vector<int> &this_bag,
                                const int pos_candidate_upper_bound = INT_MAX, 
                                const int neg_candidate_upper_bound = INT_MAX);

  // Motion Model around bbox_curr_prior_tight_
  void GetCandidates(BoundingBox &cur_bbox, int W, int H, std::vector<BoundingBox> &candidate_bboxes);

  // Check if generated candidate is valid or not
  bool ValidCandidate(BoundingBox &candidate_bbox, int W, int H);

  // Get one moved box, according to the Gaussian Motion Model
  BoundingBox GenerateOneGaussianCandidate(int W, int H, BoundingBox &bbox);

  // Create and Enqueue Training Samples given already set up example_generator
  virtual void EnqueueOnlineTraningSamples(ExampleGenerator* example_generator, const cv::Mat &image_curr, const BoundingBox &estimate,  bool success_frame);

  // check if the current estimate is success, needed as flag to pass to EnqueueOnlineTraningSamples
  virtual bool IsSuccessEstimate();

  // clear all the related storage for tracking net video
  virtual void Reset();

private:
  gsl_rng *rng_;
  // this prediction scores for candidates
  vector<float> candidate_probabilities_;
  vector<BoundingBox> candidates_bboxes_;
  vector<int> sorted_idxes_; // the sorted indexes of probabilities from high to low

  // samples collected along each frame
  std::vector<BoundingBox> gts_;
  std::vector<cv::Mat> images_finetune_;
  std::vector<cv::Mat> targets_finetune_;
  std::vector<std::vector<cv::Mat> > candidates_finetune_pos_;
  std::vector<std::vector<cv::Mat> > candidates_finetune_neg_; 
  // std::vector<std::vector<double> > labels_finetune_pos_;
  // std::vector<std::vector<double> > labels_finetune_neg_;

  // long term and short term 
  std::vector<int> short_term_bag_;
  std::vector<int> long_term_bag_;

};

#endif