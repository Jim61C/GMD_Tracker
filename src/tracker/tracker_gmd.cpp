#include "tracker_gmd.h"
#include <opencv2/videostab/inpainting.hpp>

#include "helper/helper.h"
#include "helper/bounding_box.h"
#include "network/regressor.h"
#include "helper/high_res_timer.h"
#include "helper/image_proc.h"
#include <algorithm>    // std::min

#define DEBUG_SHOW_CANDIDATES

TrackerGMD::TrackerGMD(const bool show_tracking) :
    Tracker(show_tracking)
{
    gsl_rng_env_setup();
    rng_ = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng_, time(NULL));
}

// Estimate the location of the target object in the current image.
void TrackerGMD::Track(const cv::Mat& image_curr, RegressorBase* regressor, BoundingBox* bbox_estimate_uncentered) {
    // Get target from previous image.
    cv::Mat target_pad;
    CropPadImage(bbox_prev_tight_, image_prev_, &target_pad);

    // Crop the current image based on predicted prior location of target.
    cv::Mat curr_search_region;
    BoundingBox search_location;
    double edge_spacing_x, edge_spacing_y;
    CropPadImage(bbox_curr_prior_tight_, image_curr, &curr_search_region, &search_location, &edge_spacing_x, &edge_spacing_y);

    // Motion Model to get candidate_bboxes
    std::vector<BoundingBox> candidate_bboxes;
    GetCandidates(bbox_curr_prior_tight_, image_curr.size().width, image_curr.size().height, candidate_bboxes);

    vector<float> positive_probabilities;
    vector<int> sorted_idxes; // sorted indexes of candidates from highest positive prob to lowest
    // Estimate the bounding box location as the ML estimate of the candidate_bboxes
    regressor->Predict(image_curr, curr_search_region, target_pad, candidate_bboxes, bbox_estimate_uncentered, &positive_probabilities, &sorted_idxes);

#ifdef DEBUG_SHOW_CANDIDATES

    double max_w = 0;
    double min_w = 1.0;

    for (int i =0;i< positive_probabilities.size(); i++) {
        if (positive_probabilities[i] > max_w) {
            max_w = positive_probabilities[i];
        }
        if (positive_probabilities[i] < min_w) {
            min_w = positive_probabilities[i];
        }
    }

    int min_w_color = 60;
    int max_w_color = 255;

    Mat image_to_show = image_curr.clone();
    for (int i = 0;i < candidate_bboxes.size(); i++) {
        float this_color = (int)((positive_probabilities[i]- min_w)/(max_w - min_w) * (max_w_color - min_w_color) + min_w_color);
        candidate_bboxes[i].Draw(125, 125, this_color,
                       &image_to_show);
    }
    cv::imshow("candidates", image_to_show);
    cv::waitKey(1);
#endif


    // Save the image.
    image_prev_ = image_curr;

    // Save the current estimate as the location of the target.
    bbox_prev_tight_ = *bbox_estimate_uncentered;

    // Save the current estimate as the prior prediction for the next image.
    // TODO - replace with a motion model prediction?
    bbox_curr_prior_tight_ = *bbox_estimate_uncentered;
}

bool TrackerGMD::ValidCandidate(BoundingBox &candidate_bbox, int W, int H) {
    // // make sure is inside W, H
    // if (moved_bbox.x1_ < 0) {
    //     moved_bbox.x1_ = 0.0;
    // }
    // if (moved_bbox.x2_ > W - 1) {
    //     moved_bbox.x2_ = W -1.0;
    // }

    // if (moved_bbox.y1_ < 0) {
    //     moved_bbox.y1_ = 0;
    // }
    // if (moved_bbox.y2_ > H - 1) {
    //     moved_bbox.y2_ = H - 1.0;
    // }
    
    // make sure is inside W, H
    if (candidate_bbox.x1_ < 0) {
        return false;
    }
    
    if (candidate_bbox.x2_ > W - 1) {
        return false;
    }

    if (candidate_bbox.y1_ < 0) {
        return false;
    }
    
    if (candidate_bbox.y2_ > H - 1) {
        return false;
    }

    if (candidate_bbox.x2_ <= candidate_bbox.x1_) {
        return false;
    }

    if (candidate_bbox.y2_ <= candidate_bbox.y1_) {
        return false;
    }

    return true;
}

void TrackerGMD::GetCandidates(BoundingBox &cur_bbox, int W, int H, std::vector<BoundingBox> &candidate_bboxes) {
    while(candidate_bboxes.size() < SAMPLE_CANDIDATES ) {
        BoundingBox this_candidate_bbox = GenerateOneGaussianCandidate(W, H, cur_bbox);
        if (ValidCandidate(this_candidate_bbox, W, H)) {
            candidate_bboxes.push_back(this_candidate_bbox);
        }
    }
}

BoundingBox TrackerGMD::GenerateOneGaussianCandidate(int W, int H, BoundingBox &bbox) {
  double w = bbox.x2_ - bbox.x1_;
  double h = bbox.y2_ - bbox.y1_;
  
  double centre_x = bbox.x1_ + w/2.0;
  double centre_y = bbox.y1_ + h/2.0;

  double r = round((w+h)/2.0);

  double moved_centre_x = centre_x + SD_X * r * std::max(-KEEP_SD, std::min(KEEP_SD, gsl_ran_gaussian(rng_, 1.0))); // keep the range in [-KEEP_SD* SD, KEEP_SD*SD]
  double moved_centre_y = centre_y + SD_Y * r * std::max(-KEEP_SD, std::min(KEEP_SD, gsl_ran_gaussian(rng_, 1.0))); 

  double ds = pow(MOTION_SCALE_FACTOR, SD_SCALE * std::max(-KEEP_SD, std::min(KEEP_SD, gsl_ran_gaussian(rng_, 1.0))) );
  double moved_w = w * ds;
  double moved_h = h * ds;

  BoundingBox moved_bbox;
  moved_bbox.x1_ = moved_centre_x - moved_w /2.0;
  moved_bbox.y1_ = moved_centre_y - moved_h /2.0;
  moved_bbox.x2_ = moved_centre_x + moved_w/2.0;
  moved_bbox.y2_ = moved_centre_y + moved_h/2.0;
  
  return moved_bbox;
}

void TrackerGMD::FineTuneWorker(ExampleGenerator* example_generator,
                                RegressorTrainBase* regressor_train) {
    // Actually perform fine tuning, note that do not do data augmentation for GOTURN part here

}

void TrackerGMD::FineTuneOnline(size_t frame_num, ExampleGenerator* example_generator,
                                RegressorTrainBase* regressor_train) {
    // check if to fine tune or not

}

void TrackerGMD::EnqueueOnlineTraningSamples(ExampleGenerator* example_generator) {
    std::vector<cv::Mat> this_frame_candidates;
    std::vector<double> this_frame_labels;
    example_generator->MakeCandidatesAndLabels(&this_frame_candidates, &this_frame_labels);
    candidates_finetune_.push_back(this_frame_candidates);
    labels_finetune_.push_back(this_frame_labels);

    cv::Mat image;
    cv::Mat target;
    BoundingBox bbox_gt_scaled;
    example_generator->MakeTrueExample(&image, &target, &bbox_gt_scaled);
    images_finetune_.push_back(image);
    targets_finetune_.push_back(target);
}