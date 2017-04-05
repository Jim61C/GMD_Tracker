#include "tracker_gmd.h"
#include <opencv2/videostab/inpainting.hpp>

#include "helper/helper.h"
#include "helper/bounding_box.h"
#include "network/regressor.h"
#include "helper/high_res_timer.h"
#include "helper/image_proc.h"
#include <algorithm>    // std::min

// #define DEBUG_SHOW_CANDIDATES
#define DEBUG_FINETUNE_WORKER

TrackerGMD::TrackerGMD(const bool show_tracking) :
    Tracker(show_tracking)
{
    gsl_rng_env_setup();
    rng_ = gsl_rng_alloc(gsl_rng_mt19937);
    // gsl_rng_set(rng_, time(NULL));
    gsl_rng_set(rng_, SEED_RNG_TRACKER); // to reproduce
    // engine_.seed(time(NULL));
    engine_.seed(SEED_ENGINE);
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

    // Motion Model to get candidate_bboxes, use class attributes, record the scores and candidates
    candidates_bboxes_.clear();
    GetCandidates(bbox_curr_prior_tight_, image_curr.size().width, image_curr.size().height, candidates_bboxes_);

    candidate_probabilities_.clear();
    sorted_idxes_.clear(); // sorted indexes of candidates from highest positive prob to lowest
    // Estimate the bounding box location as the ML estimate of the candidate_bboxes
    regressor->Predict(image_curr, curr_search_region, target_pad, candidates_bboxes_, bbox_estimate_uncentered, &candidate_probabilities_, &sorted_idxes_);

#ifdef DEBUG_SHOW_CANDIDATES

    double max_w = 0;
    double min_w = 1.0;

    for (int i =0;i< candidate_probabilities_.size(); i++) {
        if (candidate_probabilities_[i] > max_w) {
            max_w = candidate_probabilities_[i];
        }
        if (candidate_probabilities_[i] < min_w) {
            min_w = candidate_probabilities_[i];
        }
    }

    int min_w_color = 60;
    int max_w_color = 255;

    Mat image_to_show = image_curr.clone();
    for (int i = 0;i < candidates_bboxes_.size(); i++) {
        float this_color = (int)((candidate_probabilities_[i]- min_w)/(max_w - min_w) * (max_w_color - min_w_color) + min_w_color);
        candidates_bboxes_[i].Draw(125, 125, this_color,
                       &image_to_show);
    }
    cv::imshow("candidates", image_to_show);
    cv::waitKey(1);
#endif

}

bool TrackerGMD::ValidCandidate(BoundingBox &candidate_bbox, int W, int H) {
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
        // // crop against W, H so that fit in image
        // this_candidate_bbox.crop_against_width_height(W, H);
        // if at boarder, do not crop, to avoid really thin candidates being fed in -> correspond to cropping gt in training, instead of cropping pos/neg samples
        if (this_candidate_bbox.valid_bbox_against_width_height(W, H)) {
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
                                RegressorTrainBase* regressor_train,
                                vector<int> &this_bag,
                                const int pos_candidate_upper_bound, 
                                const int neg_candidate_upper_bound) {

    vector<int> this_bag_permuted(this_bag);
    std::shuffle(this_bag_permuted.begin(), this_bag_permuted.end(), engine_);

    // Actually perform fine tuning, note that do not do data augmentation for GOTURN part here
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> targets;
    std::vector<BoundingBox> bboxes_gt_scaled;
    std::vector<std::vector<cv::Mat> > candidates; 
    std::vector<std::vector<double> >  labels;
    
    for (int i = 0; i< this_bag_permuted.size(); i ++) {
        vector<pair<double, Mat> > label_to_candidate;
        vector<double> this_frame_labels;
        vector<Mat> this_frame_candidates;
        
        int this_update_idx = this_bag_permuted[i];
        for (int j = 0; j < std::min((int)(candidates_finetune_pos_[this_update_idx].size()), pos_candidate_upper_bound);j++) {
            label_to_candidate.push_back(std::make_pair(POS_LABEL, candidates_finetune_pos_[this_update_idx][j]));
        }
        for (int j = 0; j < std::min((int)(candidates_finetune_neg_[this_update_idx].size()), neg_candidate_upper_bound);j++) {
            label_to_candidate.push_back(std::make_pair(NEG_LABEL, candidates_finetune_neg_[this_update_idx][j]));
        }

        // random shuffle
        std::shuffle(std::begin(label_to_candidate), std::end(label_to_candidate), engine_);

        for (int i = 0; i< label_to_candidate.size(); i++) {
            this_frame_candidates.push_back(label_to_candidate[i].second);
            this_frame_labels.push_back(label_to_candidate[i].first);
        }


        images.push_back(images_finetune_[this_update_idx]);
        targets.push_back(targets_finetune_[this_update_idx]);
        bboxes_gt_scaled.push_back(gts_[this_update_idx]);
        candidates.push_back(this_frame_candidates);
        labels.push_back(this_frame_labels);
    }

#ifdef DEBUG_FINETUNE_WORKER
    int count = 0;
    for (int i = 0; i< candidates.size();i++) {
        count += candidates[i].size();
    }
    cout << "Total number of candidates for fine tune: " << count << endl;
#endif

    // feed to network to train
    regressor_train->TrainBatch(images,
                            targets,
                            bboxes_gt_scaled,
                            candidates,
                            labels,
                            -1); // -1 indicating fine tuning

}

void TrackerGMD::FineTuneOnline(size_t frame_num, ExampleGenerator* example_generator,
                                RegressorTrainBase* regressor_train, bool success_frame, bool is_last_frame) {
    // check if to fine tune or not
    // check if need long term finetune
    if (cur_frame_ != 0 && !is_last_frame && (cur_frame_ % LONG_TERM_UPDATE_INTERVAL == 0)) {
        cout << "cur_frame_:" << cur_frame_ << ", about to start long term fine tune, frames to use:" << endl;
        for (int i = 0 ; i < long_term_bag_.size(); i ++ ) {
            cout << long_term_bag_[i] << ", ";
        }
        cout << endl;
        FineTuneWorker(example_generator,
                       regressor_train,
                       long_term_bag_,
                       LONG_TERM_CANDIDATE_UPPER_BOUND, 
                       LONG_TERM_CANDIDATE_UPPER_BOUND);
    }
    


    // check if need short_term finetune, if best prob < 0.5, need to short term finetune
    if (!success_frame) {
        cout << "cur_frame_:" << cur_frame_ << ", about to start short term fine tune, frames to use:" << endl;
        for (int i = 0 ; i < short_term_bag_.size(); i ++ ) {
            cout << short_term_bag_[i] << ", ";
        }
        cout << endl;
        FineTuneWorker(example_generator,
                       regressor_train,
                       short_term_bag_);
    }

}


void TrackerGMD::EnqueueOnlineTraningSamples(ExampleGenerator* example_generator, const cv::Mat &image_curr, const BoundingBox &estimate,  bool success_frame) {

    std::vector<cv::Mat> this_frame_candidates_pos;
    std::vector<cv::Mat> this_frame_candidates_neg;

    cv::Mat image;
    cv::Mat target;
    BoundingBox bbox_gt_scaled;

    if (success_frame) {
        // reset example_generator
        example_generator->Reset(bbox_prev_tight_,
                                 estimate,
                                 image_prev_, 
                                 image_curr);
        
        cout << "cur_frame_:" << cur_frame_<< " is success frame, enqueue pos and neg examples for later fine tuning" << endl;
        example_generator->MakeCandidatesPos(&this_frame_candidates_pos);
        example_generator->MakeCandidatesNeg(&this_frame_candidates_neg, NEG_CANDIDATES/2);
        example_generator->MakeCandidatesNeg(&this_frame_candidates_neg, NEG_CANDIDATES/2, NEG_TRANS_RANGE, NEG_SCALE_RANGE, "whole");
        example_generator->MakeTrueExample(&image, &target, &bbox_gt_scaled);

        // enqueue this frame index
        short_term_bag_.push_back(cur_frame_);
        long_term_bag_.push_back(cur_frame_);

        // remove old frames kept for online update
        while(short_term_bag_.size() > SHORT_TERM_BAG_SIZE) {
            // no removing here as the frames could still be used for long term fine tuning, only remove when it even goes out of long term range
            short_term_bag_.erase(short_term_bag_.begin());
        }

        while(long_term_bag_.size() > LONG_TERM_BAG_SIZE) {
            // remove from beginning
            candidates_finetune_pos_[long_term_bag_[0]].clear();
            candidates_finetune_neg_[long_term_bag_[0]].clear();
            long_term_bag_.erase(long_term_bag_.begin());
        }
    }

    // if not success, push back dummy values
    candidates_finetune_pos_.push_back(this_frame_candidates_pos);
    candidates_finetune_neg_.push_back(this_frame_candidates_neg); 

    images_finetune_.push_back(image);
    targets_finetune_.push_back(target);
    gts_.push_back(bbox_gt_scaled);
}

struct MyGreater
{
    template<class T>
    bool operator()(T const &a, T const &b) const { return a > b; }
};


bool TrackerGMD::IsSuccessEstimate() {

  double prob_sum = 0;
  // get top 5 score average 
  for (int i = 0 ; i< TOP_ESTIMATES; i ++) {
    double this_prob = candidate_probabilities_[sorted_idxes_[i]];
    prob_sum += this_prob;
  }

  double avg_prob = prob_sum / TOP_ESTIMATES;
  cout <<"cur_frame_:" << cur_frame_ << " avg_prob: " << avg_prob << endl;
  if (avg_prob <= SHORT_TERM_FINE_TUNE_TH) {
      return false;
  }
  else {
      return true;
  }

}

void TrackerGMD::Reset() {
    cur_frame_ = 0;
    candidate_probabilities_.clear();
    candidate_probabilities_.reserve(SAMPLE_CANDIDATES);
    candidates_bboxes_.clear();
    candidates_bboxes_.reserve(SAMPLE_CANDIDATES);
    sorted_idxes_.clear();
    sorted_idxes_.reserve(SAMPLE_CANDIDATES);

    // samples collected along each frame
    gts_.clear();
    images_finetune_.clear();
    targets_finetune_.clear();
    candidates_finetune_pos_.clear();
    candidates_finetune_neg_.clear(); 

    // long term and short term 
    short_term_bag_.clear();
    long_term_bag_.clear();
}