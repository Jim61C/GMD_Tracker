#ifndef TRACKER_GMD_H
#define TRACKER_GMD_H

#include "tracker.h"

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <gsl/gsl_rng.h> 
#include <gsl/gsl_randist.h> /* GAUSSIAN*/
#include "helper/Constants.h"
#include <limits.h>
#include "helper/high_res_timer.h"

class TrackerGMD : public Tracker {

public:
  TrackerGMD(const bool show_tracking, ExampleGenerator* example_generator,  RegressorTrainBase* regressor_train);

  // Estimate the location of the target object in the current image.
  virtual void Track(const cv::Mat& image_curr, RegressorBase* regressor,
             BoundingBox* bbox_estimate_uncentered);

  // After tracking for this frame, update internal state
  virtual void UpdateState(const cv::Mat& image_curr, BoundingBox &bbox_estimate, RegressorBase* regressor, bool is_last_frame);

  // Initialize the tracker with the ground-truth bounding box of the first frame.
  virtual void Init(const cv::Mat& image_curr, const BoundingBox& bbox_gt,
            RegressorBase* regressor);

  virtual void Init(const std::string& image_curr_path, const VOTRegion& region, 
            RegressorBase* regressor);
  
  // Online fine tune, given the networks and example_generators
  virtual void FineTuneOnline(ExampleGenerator* example_generator,
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
  BoundingBox GenerateOneGaussianCandidate(int W, int H, BoundingBox &bbox, double sd_x = SD_X, double sd_y = SD_Y, 
                                           double sd_scale = SD_SCALE, double sd_ap = SD_AP);

  // Create and Enqueue Training Samples given already set up example_generator
  virtual void EnqueueOnlineTraningSamples(ExampleGenerator* example_generator, const cv::Mat &image_curr, const BoundingBox &estimate,  bool success_frame);

  // check if the current estimate is success, needed as flag to pass to EnqueueOnlineTraningSamples
  virtual bool IsSuccessEstimate();

  // clear all the related storage for tracking net video
  virtual void Reset(RegressorBase *regressor);

private:
  gsl_rng *rng_;

  // Used to generate additional training examples through synthetic transformations.
  ExampleGenerator* example_generator_;

  // Neural network.
  RegressorTrainBase* regressor_train_;


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

  std::mt19937 engine_;

  // for motion model candidates
  double sd_trans_;
  double sd_scale_;
  double sd_ap_;

  // timer
  HighResTimer hrt_;

};

#endif