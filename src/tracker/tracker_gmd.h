#ifndef TRACKER_GMD_H
#define TRACKER_GMD_H

#include "tracker.h"

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <gsl/gsl_rng.h> 
#include <gsl/gsl_randist.h> /* GAUSSIAN*/

#define SAMPLE_CANDIDATES 250
#define SD_X 0.3 // translation std: mean(width,height)*SD_X
#define SD_Y 0.3
#define SD_SCALE 0.5 
#define KEEP_SD 2.0
#define MOTION_SCALE_FACTOR 1.05 // scaling std: MOTION_SCALE_FACTOR^(SD_SCALE)

class TrackerGMD : public Tracker {

public:
  TrackerGMD(const bool show_tracking);

  // Estimate the location of the target object in the current image.
  virtual void Track(const cv::Mat& image_curr, RegressorBase* regressor,
             BoundingBox* bbox_estimate_uncentered);

  // Motion Model around bbox_curr_prior_tight_
  void GetCandidates(BoundingBox &cur_bbox, int W, int H, std::vector<BoundingBox> &candidate_bboxes);

  // Get one moved box, according to the Gaussian Motion Model
  BoundingBox GenerateOneGaussianCandidate(int W, int H, BoundingBox &bbox);

private:
  gsl_rng *rng_;
};

#endif