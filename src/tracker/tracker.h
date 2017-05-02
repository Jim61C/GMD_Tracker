#ifndef TRACKER_H
#define TRACKER_H

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "helper/bounding_box.h"
#include "train/example_generator.h"
#include "network/regressor.h"
#include "network/regressor_train_base.h"

class Tracker
{
public:
  Tracker(const bool show_tracking);

  // Estimate the location of the target object in the current image.
  virtual void Track(const cv::Mat& image_curr, RegressorBase* regressor,
             BoundingBox* bbox_estimate_uncentered);

  // After tracking for this frame, update internal state
  virtual void UpdateState(const cv::Mat& image_curr, BoundingBox &bbox_estimate, RegressorBase* regressor, bool is_last_frame);

  // Initialize the tracker with the ground-truth bounding box of the first frame.
  virtual void Init(const cv::Mat& image_curr, const BoundingBox& bbox_gt,
            RegressorBase* regressor);

  // Initialize the tracker with the ground-truth bounding box of the first frame.
  // VOTRegion is an object for initializing the tracker when using the VOT Tracking dataset.
  virtual void Init(const std::string& image_curr_path, const VOTRegion& region,
            RegressorBase* regressor);
  
  
  // Online fine tune, given the networks and example_generators
  virtual void FineTuneOnline(ExampleGenerator* example_generator,
                                RegressorTrainBase* regressor_train, bool success_frame, bool is_last_frame);
  
  // Create and Enqueue Training Samples given already set up example_generator
  virtual void EnqueueOnlineTraningSamples(ExampleGenerator* example_generator, const cv::Mat &image_curr, const BoundingBox &estimate,  bool success_frame) { }

  // check if the current estimate is success, needed as flag to pass to EnqueueOnlineTraningSamples
  virtual bool IsSuccessEstimate() { return true; }

  // clear all the storage associated with this video for next video
  virtual void Reset(RegressorBase *regressor) { }

  cv::Mat GetImagePrev() { return image_prev_; }

  void SetImagePrev(cv::Mat image_prev) { image_prev_ = image_prev; }

  BoundingBox GetBBoxPrev() { return bbox_prev_tight_; }

  void SetBBoxPrev(BoundingBox bbox) { bbox_prev_tight_ = bbox; }

  int GetInternelCurFrame() { return cur_frame_; }

protected:
  // Show the tracking output, for debugging.
  void ShowTracking(const cv::Mat& target_pad, const cv::Mat& curr_search_region, const BoundingBox& bbox_estimate) const;

  // Predicted prior location of the target object in the current image.
  // This should be a tight (high-confidence) prior prediction area.  We will
  // add padding to this region.
  BoundingBox bbox_curr_prior_tight_;

  // Estimated previous location of the target object.
  BoundingBox bbox_prev_tight_;

  // Full previous image.
  cv::Mat image_prev_;

  // Whether to visualize the tracking results
  bool show_tracking_;

  // internel index starting from 0
  int cur_frame_;
};

#endif // TRACKER_H
