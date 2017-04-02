#ifndef TRACKER_GMD_H
#define TRACKER_GMD_H

#include "tracker.h"

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <gsl/gsl_rng.h> 
#include <gsl/gsl_randist.h> /* GAUSSIAN*/
#include "helper/Constants.h"

class TrackerGMD : public Tracker {

public:
  TrackerGMD(const bool show_tracking);

  // Estimate the location of the target object in the current image.
  virtual void Track(const cv::Mat& image_curr, RegressorBase* regressor,
             BoundingBox* bbox_estimate_uncentered);
  
  // Online fine tune, given the networks and example_generators
  virtual void FineTuneOnline(size_t frame_num, ExampleGenerator* example_generator,
                                RegressorTrainBase* regressor_train);
  
  // Actual worker to do the finetune
  void FineTuneWorker(ExampleGenerator* example_generator,
                                RegressorTrainBase* regressor_train);

  // Motion Model around bbox_curr_prior_tight_
  void GetCandidates(BoundingBox &cur_bbox, int W, int H, std::vector<BoundingBox> &candidate_bboxes);

  // Check if generated candidate is valid or not
  bool ValidCandidate(BoundingBox &candidate_bbox, int W, int H);

  // Get one moved box, according to the Gaussian Motion Model
  BoundingBox GenerateOneGaussianCandidate(int W, int H, BoundingBox &bbox);

  // Create and Enqueue Training Samples given already set up example_generator
  void EnqueueOnlineTraningSamples(ExampleGenerator* example_generator);

private:
  gsl_rng *rng_;
  // this prediction scores for candidates
  vector<float> candidate_scores_;
  vector<BoundingBox> candidates_;

  // samples collected along
  std::vector<cv::Mat> images_finetune_;
  std::vector<cv::Mat> targets_finetune_;
  std::vector<std::vector<cv::Mat> > candidates_finetune_; 
  std::vector<std::vector<double> > labels_finetune_;

};

#endif