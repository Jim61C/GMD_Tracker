#include "tracker_gmd.h"
#include <opencv2/videostab/inpainting.hpp>

#include "helper/helper.h"
#include "helper/bounding_box.h"
#include "network/regressor.h"
#include "helper/high_res_timer.h"
#include "helper/image_proc.h"
#include <algorithm>    // std::min

// #define DEBUG_SHOW_CANDIDATES
// // #define DEBUG_FINETUNE_WORKER
// #define FISRT_FRAME_PAUSE
// // #define VISUALIZE_FIRST_FRAME_SAMPLES
// #define DEBUG_LOG
// // #define LOG_TIME

TrackerGMD::TrackerGMD(const bool show_tracking, ExampleGenerator* example_generator,  RegressorTrainBase* regressor_train) :
    Tracker(show_tracking),
    example_generator_(example_generator),
    regressor_train_(regressor_train),
    hrt_("TrackerGMD")
{
    gsl_rng_env_setup();
    rng_ = gsl_rng_alloc(gsl_rng_mt19937);
    // gsl_rng_set(rng_, time(NULL));
    gsl_rng_set(rng_, SEED_RNG_TRACKER); // to reproduce
    // engine_.seed(time(NULL));
    engine_.seed(SEED_ENGINE);

    sd_trans_ = SD_X;
    sd_scale_ = SD_SCALE;
    sd_ap_ = SD_AP;
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

    // get target_tight
    cv::Mat target_tight;
    BoundingBox bbox_prev_within(bbox_prev_tight_);
    bbox_prev_within.crop_against_width_height(image_prev_.size().width, image_prev_.size().height);
    bbox_prev_within.CropBoundingBoxOutImage(image_prev_, &target_tight);

    // Motion Model to get candidate_bboxes, use class attributes, record the scores and candidates
    candidates_bboxes_.clear();

#ifdef LOG_TIME
    hrt_.reset();
    hrt_.start();
#endif
    GetCandidates(bbox_curr_prior_tight_, image_curr.size().width, image_curr.size().height, candidates_bboxes_);
#ifdef LOG_TIME
    hrt_.stop();
    cout << "time spent for genrating motion candiadates: " << hrt_.getMilliseconds() << " ms" << endl;
#endif

    candidate_probabilities_.clear();
    sorted_idxes_.clear(); // sorted indexes of candidates from highest positive prob to lowest
    // Estimate the bounding box location as the ML estimate of the candidate_bboxes
    regressor->PredictFast(image_curr, curr_search_region, target_tight, candidates_bboxes_, bbox_prev_tight_, bbox_estimate_uncentered, &candidate_probabilities_, &sorted_idxes_, sd_trans_, cur_frame_);

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

    // show the top few
    int top_num = 5;
    double top_few_min = 1.0;
    double top_few_max = 0.0;
    for (int i =0 ; i < top_num; i ++) {
        if (candidate_probabilities_[sorted_idxes_[i]] > top_few_max) {
            top_few_max = candidate_probabilities_[sorted_idxes_[i]];
        }
        if (candidate_probabilities_[sorted_idxes_[i]] < top_few_min) {
            top_few_min = candidate_probabilities_[sorted_idxes_[i]];
        }
    }

    Mat image_top_few = image_curr.clone();
    min_w_color = 0;
    max_w_color = 255;
    for (int i = 0; i < top_num; i ++) {
        float this_color = (int)((candidate_probabilities_[sorted_idxes_[i]] - top_few_min)/ (top_few_max - top_few_min) * (max_w_color - min_w_color) + min_w_color);
        candidates_bboxes_[sorted_idxes_[i]].Draw(this_color, 0, 0, &image_top_few);
        cv::putText(image_top_few, "box" + std::to_string(sorted_idxes_[i]) + ":" + std::to_string(candidate_probabilities_[sorted_idxes_[i]]),
            cv::Point(candidates_bboxes_[sorted_idxes_[i]].get_center_x(), candidates_bboxes_[sorted_idxes_[i]].get_center_y()), 
            FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    imshow("top few candidates used for estimation", image_top_few);
    waitKey(1);
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
        BoundingBox this_candidate_bbox = GenerateOneGaussianCandidate(W, H, cur_bbox, sd_trans_, sd_trans_, sd_scale_, sd_ap_);
        // // crop against W, H so that fit in image
        // this_candidate_bbox.crop_against_width_height(W, H);
        // if at boarder, do not crop, to avoid really thin candidates being fed in -> correspond to cropping gt in training, instead of cropping pos/neg samples
        if (this_candidate_bbox.valid_bbox_against_width_height(W, H)) {
            candidate_bboxes.push_back(this_candidate_bbox);
        }
    }
}

BoundingBox TrackerGMD::GenerateOneGaussianCandidate(int W, int H, BoundingBox &bbox, double sd_x, double sd_y, double sd_scale, double sd_ap) {
  double w = bbox.x2_ - bbox.x1_;
  double h = bbox.y2_ - bbox.y1_;
  double ap = h/w;
  
  double centre_x = bbox.x1_ + w/2.0;
  double centre_y = bbox.y1_ + h/2.0;

  double r = round((w+h)/2.0);

  double moved_centre_x = centre_x + sd_x * r * std::max(-KEEP_SD, std::min(KEEP_SD, gsl_ran_gaussian(rng_, 1.0))); // keep the range in [-KEEP_SD* SD, KEEP_SD*SD]
  double moved_centre_y = centre_y + sd_y * r * std::max(-KEEP_SD, std::min(KEEP_SD, gsl_ran_gaussian(rng_, 1.0))); 

  double ds = pow(MOTION_SCALE_FACTOR, sd_scale * std::max(-KEEP_SD, std::min(KEEP_SD, gsl_ran_gaussian(rng_, 1.0))) );
  double dap = pow(MOTION_AP_FACTOR, sd_ap * std::max(-KEEP_SD, std::min(KEEP_SD, gsl_ran_gaussian(rng_, 1.0))) );
  double moved_w = w * ds;
  double moved_h = moved_w * (ap * dap);

  BoundingBox moved_bbox;
  moved_bbox.x1_ = moved_centre_x - moved_w /2.0;
  moved_bbox.y1_ = moved_centre_y - moved_h /2.0;
  moved_bbox.x2_ = moved_centre_x + moved_w/2.0;
  moved_bbox.y2_ = moved_centre_y + moved_h/2.0;
  
  return moved_bbox;
}

void TrackerGMD::FineTuneWorker(ExampleGenerator* example_generator,
                                RegressorTrainBase* regressor_train,
                                std::vector<int> &this_bag,
                                const int pos_candidate_upper_bound, 
                                const int neg_candidate_upper_bound) {

    std::vector<int> this_bag_permuted(this_bag);
    std::shuffle(this_bag_permuted.begin(), this_bag_permuted.end(), engine_);

    // Actually perform fine tuning, note that do not do data augmentation for GOTURN part here
    std::vector<cv::Mat> image_currs;
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> targets;
    std::vector<BoundingBox> bboxes_gt_scaled;
    std::vector<std::vector<BoundingBox> > candidates; 
    std::vector<std::vector<double> >  labels;
    
    for (int i = 0; i< this_bag_permuted.size(); i ++) {
        std::vector<pair<double, BoundingBox> > label_to_candidate;
        std::vector<double> this_frame_labels;
        std::vector<BoundingBox> this_frame_candidates;
        
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

        image_currs.push_back(image_currs_[this_update_idx]);
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
    // regressor_train->TrainBatchFast(image_currs,
    //                         images,
    //                         targets,
    //                         bboxes_gt_scaled,
    //                         candidates,
    //                         labels,
    //                         -1,
    //                         pos_candidate_upper_bound + neg_candidate_upper_bound, 
    //                         NOHEM_FINETUNE); // k == -1 indicating fine tuning
    
    regressor_train->TrainBatchFast(image_currs,
                            images,
                            targets,
                            bboxes_gt_scaled,
                            candidates,
                            labels,
                            -1); // k == -1 indicating fine tuning

}

void TrackerGMD::FineTuneOnline(ExampleGenerator* example_generator,
                                RegressorTrainBase* regressor_train, bool success_frame, bool is_last_frame) {
    // check if to fine tune or not
    // check if need long term finetune
    if (cur_frame_ != 0 && !is_last_frame && (cur_frame_ % LONG_TERM_UPDATE_INTERVAL == 0)) {
#ifdef DEBUG_LOG
        cout << "cur_frame_:" << cur_frame_ << ", about to start long term fine tune, frames to use:" << endl;
        for (int i = 0 ; i < long_term_bag_.size(); i ++ ) {
            cout << long_term_bag_[i] << ", ";
        }
        cout << endl;
#endif
        FineTuneWorker(example_generator,
                       regressor_train,
                       long_term_bag_,
                       LONG_TERM_POS_CANDIDATE_UPPER_BOUND, 
                       LONG_TERM_NEG_CANDIDATE_UPPER_BOUND);
    }
    


    // check if need short_term finetune, if best prob < 0.5, need to short term finetune
    if (!success_frame) {
#ifdef DEBUG_LOG
        cout << "cur_frame_:" << cur_frame_ << ", about to start short term fine tune, frames to use:" << endl;
        for (int i = 0 ; i < short_term_bag_.size(); i ++ ) {
            cout << short_term_bag_[i] << ", ";
        }
        cout << endl;
#endif
        FineTuneWorker(example_generator,
                       regressor_train,
                       short_term_bag_);
    }

}


void TrackerGMD::EnqueueOnlineTraningSamples(ExampleGenerator* example_generator, const cv::Mat &image_curr, const BoundingBox &estimate,  bool success_frame) {

    std::vector<BoundingBox> this_frame_candidates_pos;
    std::vector<BoundingBox> this_frame_candidates_neg;

    cv::Mat image;
    cv::Mat target;
    BoundingBox bbox_gt_scaled;

    if (success_frame) {
        // reset example_generator
        example_generator->Reset(bbox_prev_tight_,
                                 estimate,
                                 image_prev_, 
                                 image_curr);
#ifdef DEBUG_LOG
        cout << "cur_frame_:" << cur_frame_<< " is success frame, enqueue pos and neg examples for later fine tuning" << endl;
#endif
        example_generator->MakeCandidatesPos(&this_frame_candidates_pos, POS_CANDIDATES_FINETUNE, "gaussian", POS_TRANS_RANGE, POS_SCALE_RANGE, 
                                             0.05, 0.05, 2.5); // trans sd 0.05, scale sd 2.5
        example_generator->MakeCandidatesNeg(&this_frame_candidates_neg, NEG_CANDIDATES_FINETUNE, "uniform", 1.0, 2.5); // trans range 1, scale range 2.5
        // example_generator->MakeCandidatesNeg(&this_frame_candidates_neg, NEG_CANDIDATES_FINETUNE/2, "whole", NEG_TRANS_RANGE, 5.0);
        example_generator->MakeTrueExampleTight(&image, &target, &bbox_gt_scaled);

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

    image_currs_.push_back(image_curr);
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
#ifdef DEBUG_LOG
  cout <<"cur_frame_:" << cur_frame_ << " avg_prob: " << avg_prob << endl;
#endif
  if (avg_prob <= SHORT_TERM_FINE_TUNE_TH) {
      return false;
  }
  else {
      return true;
  }

}

void TrackerGMD::Reset(RegressorBase *regressor) {
    // Reset the fine-tuned net for next video
    regressor->Reset(); // reinitialise net_ and load new weights

    regressor_train_->ResetSolverNet(); // release solver_'s net_ memory and re-assign net_ to solver_

    cur_frame_ = 0;
    candidate_probabilities_.clear();
    candidate_probabilities_.reserve(SAMPLE_CANDIDATES);
    candidates_bboxes_.clear();
    candidates_bboxes_.reserve(SAMPLE_CANDIDATES);
    sorted_idxes_.clear();
    sorted_idxes_.reserve(SAMPLE_CANDIDATES);

    // samples collected along each frame
    gts_.clear();
    image_currs_.clear();
    images_finetune_.clear();
    targets_finetune_.clear();
    candidates_finetune_pos_.clear();
    candidates_finetune_neg_.clear(); 

    // long term and short term 
    short_term_bag_.clear();
    long_term_bag_.clear();
}


  // Initialize the tracker with the ground-truth bounding box of the first frame.
void TrackerGMD::Init(const cv::Mat& image_curr, const BoundingBox& bbox_gt, 
                      RegressorBase* regressor) {
    // Initialize the neural network.
    regressor->Init();

    // fine tune at cur_frame_ 0
    cur_frame_ = 0;

    // Train a Bbox regressor
    // TODO: the following is not needed, here just to comply with API of GetBBoxConvFeatures
    example_generator_->Reset(bbox_gt,
                        bbox_gt,
                        image_curr,
                        image_curr); // use the same image as initial step fine-tuning
    // Generate true example.
    cv::Mat image_regress;
    cv::Mat target_regress;
    BoundingBox bbox_gt_scaled_regress;
    example_generator_->MakeTrueExampleTight(&image_regress, &target_regress, &bbox_gt_scaled_regress);
    // Get the bbox conv features for training
    std::vector<BoundingBox> regress_bboxes;
    example_generator_->MakeCandidatesPos(&regress_bboxes, 1000, "uniform", POS_TRANS_RANGE, POS_SCALE_RANGE);
    std::vector<std::vector<float> > features;
    regressor->GetBBoxConvFeatures(image_curr, image_regress, target_regress, regress_bboxes, features);
    bbox_finetuner_.trainModelUsingInitialFrameBboxes(features, regress_bboxes, bbox_gt);

    printf("About to fine tune the first frame ...\n");
    for (int iter = 0; iter < FIRST_FRAME_FINETUNE_ITERATION; iter ++) {
        printf("first frame fine tune iter %d\n", iter);
        // Set up example generator.
        example_generator_->Reset(bbox_gt,
                                bbox_gt,
                                image_curr,
                                image_curr); // use the same image as initial step fine-tuning

        // data structures to invoke fine tune
        std::vector<cv::Mat> image_currs;
        std::vector<cv::Mat> images;
        std::vector<cv::Mat> targets;
        std::vector<BoundingBox> bboxes_gt_scaled;
        std::vector<std::vector<BoundingBox> > candidates; 
        std::vector<std::vector<double> >  labels;

        // Generate true example.
        cv::Mat image;
        cv::Mat target;
        BoundingBox bbox_gt_scaled;
        example_generator_->MakeTrueExampleTight(&image, &target, &bbox_gt_scaled);
        
        image_currs.push_back(image_curr);
        images.push_back(image);
        targets.push_back(target);
        bboxes_gt_scaled.push_back(bbox_gt_scaled);
                                                
        std::vector<BoundingBox> this_frame_candidates;
        std::vector<double> this_frame_labels;

        std::vector<BoundingBox> this_frame_candidates_pos;
        std::vector<BoundingBox> this_frame_candidates_neg;

        // generate candidates and push to this_frame_candidates and this_frame_labels
        // example_generator_->MakeCandidatesAndLabels(&this_frame_candidates, &this_frame_labels, FIRST_FRAME_POS_SAMPLES, FIRST_FRAME_NEG_SAMPLES);
        example_generator_->MakeCandidatesPos(&this_frame_candidates_pos, FIRST_FRAME_POS_SAMPLES, "gaussian", POS_TRANS_RANGE, POS_SCALE_RANGE,
                                              0.05, 0.05, 2.5); // 0.05, 2.5
        example_generator_->MakeCandidatesNeg(&this_frame_candidates_neg, FIRST_FRAME_NEG_SAMPLES/2, "uniform", 0.5, 5); // 0.5, 5
        example_generator_->MakeCandidatesNeg(&this_frame_candidates_neg, FIRST_FRAME_NEG_SAMPLES/2, "whole", NEG_TRANS_RANGE, 5.0);

        // shuffling
        std::vector<std::pair<double, BoundingBox> > label_to_candidate;
        for (int i =0; i < this_frame_candidates_pos.size(); i++) {
        label_to_candidate.push_back(std::make_pair(POS_LABEL, this_frame_candidates_pos[i]));
        }
        for (int i =0; i < this_frame_candidates_neg.size(); i++) {
        label_to_candidate.push_back(std::make_pair(NEG_LABEL, this_frame_candidates_neg[i]));
        }

        // random shuffle
        std::shuffle(std::begin(label_to_candidate), std::end(label_to_candidate), engine_);

        for (int i = 0; i< label_to_candidate.size(); i++) {
            this_frame_candidates.push_back(label_to_candidate[i].second);
            this_frame_labels.push_back(label_to_candidate[i].first);
        }

    #ifdef VISUALIZE_FIRST_FRAME_SAMPLES
        Mat visualise_first_frame = image_curr.clone();
        for (int i = 0; i < this_frame_candidates.size(); i++) {
            if(this_frame_labels[i] == POS_LABEL) {
                this_frame_candidates[i].Draw(255,0,0,&visualise_first_frame);
            }
            else {
                this_frame_candidates[i].Draw(0,0,255,&visualise_first_frame);
            }
        }
        cv::imshow("first frame samples", visualise_first_frame);
        cv::waitKey(1);
    #endif

        // TODO: avoid the copying and just pass a vector of one frame's +/- candidates to train
        for(int i = 0; i< images.size(); i ++ ) {
        candidates.push_back(std::vector<BoundingBox>(this_frame_candidates)); // copy
        labels.push_back(std::vector<double>(this_frame_labels)); // copy
        }

        //Fine Tune!
        regressor_train_->TrainBatchFast(image_currs,
                                    images,
                                    targets,
                                    bboxes_gt_scaled,
                                    candidates,
                                    labels,
                                    -1,
                                    (FIRST_FRAME_POS_SAMPLES + FIRST_FRAME_NEG_SAMPLES)/FIRST_FRAME_NUM_MINI_BATCH, // guaranteed that at least one frame candidates
                                    FIRST_FRAME_ONHEM); // k == -1 indicating fine tuning
        
        // regressor_train_->TrainBatchFast(image_currs,
        //                             images,
        //                             targets,
        //                             bboxes_gt_scaled,
        //                             candidates,
        //                             labels,
        //                             -1); // k == -1 indicating fine tuning

    }
    printf("Fine tune the first frame completed!\n");

#ifdef FISRT_FRAME_PAUSE
  cv::Mat image_curr_show = image_curr.clone();
  bbox_gt.DrawBoundingBox(&image_curr_show);
  cv::imshow("Full output", image_curr_show);
  cv::waitKey(0);
#endif

    // the same part as usual tracker, set the image_prev_, bbox_curr_prior_tight_, bbox_prev_tight_
    image_prev_ = image_curr;
    bbox_prev_tight_ = bbox_gt;

    // Predict in the current frame that the location will be approximately the same
    // as in the previous frame.
    // TODO - use a motion model?
    bbox_curr_prior_tight_ = bbox_gt;

    // enqueue short term online learning samples, 50 POS and 200 NEG
    EnqueueOnlineTraningSamples(example_generator_, image_curr, bbox_gt, true); // TODO, if first frame add random purturbations like GOTURN to simulate frame -1 to frame 0

    // at this point of time, cur_frame_ should be 1, since we start on 2nd frame of the sequnce, 1st frame we have the ground truth
    cur_frame_ = 1;
}

void TrackerGMD::Init(const std::string& image_curr_path, const VOTRegion& region, 
            RegressorBase* regressor) { 
    Tracker::Init(image_curr_path, region, regressor); 
}

// After tracking for this frame, update internal state, called right after tracker_->Track
void TrackerGMD::UpdateState(const cv::Mat& image_curr, BoundingBox &bbox_estimate, RegressorBase* regressor, bool is_last_frame) {
#ifdef LOG_TIME
    hrt_.reset();
    hrt_.start();
#endif
    // Post processing after this frame, fine tune, invoke tracker_ -> finetune
    bool is_this_frame_success = IsSuccessEstimate();

    // update sd_trans_ in case of failure
    if (!is_this_frame_success) {
        sd_trans_ = std::min(0.75, 1.1 * sd_trans_);
    }
    else {
        sd_trans_ = SD_X;
    }

    if (is_this_frame_success) {
        // wrap in a vector to use get features API
        std::vector<std::vector<float> > bbox_features;
        std::vector<BoundingBox> wrap_this_bbox_estimate;
        wrap_this_bbox_estimate.push_back(bbox_estimate);
        
        // TODO: the following is actually not needed, here to just comply with the API for GetBBoxConvFeatures
        example_generator_->Reset(bbox_prev_tight_,
                                  bbox_estimate,
                                  image_prev_,
                                  image_curr);
        
        // Generate true example.
        cv::Mat image_regress;
        cv::Mat target_regress;
        BoundingBox bbox_gt_scaled_regress;
        example_generator_->MakeTrueExampleTight(&image_regress, &target_regress, &bbox_gt_scaled_regress);
        
        regressor->GetBBoxConvFeatures(image_curr, image_regress, target_regress, wrap_this_bbox_estimate, bbox_features);
        bbox_finetuner_.refineBoundingBox(bbox_estimate, bbox_features[0]);
    }

    // generate examples, if not success, just dummy values pushed in
    EnqueueOnlineTraningSamples(example_generator_, image_curr, bbox_estimate, is_this_frame_success);

    // afte generate examples, check if need to fine tune, and acutally fine tune if needed 
    FineTuneOnline(example_generator_, regressor_train_, is_this_frame_success, is_last_frame);

    // TODO: check if re-estimate after failure works better
    // Track(image_curr, regressor, bbox_estimate_uncentered);

    // TODO: when appearance change drastically, after re-estimate, still enqueue for finetune
    // hypothesis: if there is a drastic drop in target score, indiating appearance change, need to enqueu and finetune!

    // update image_prev_ to image_curr
    image_prev_ = image_curr;

    // Save the current estimate as the location of the target.
    bbox_prev_tight_ = bbox_estimate;

    // Save the current estimate as the prior prediction for the next image.
    // TODO - replace with a motion model prediction?
    bbox_curr_prior_tight_ = bbox_estimate;

    // internel frame counter
    cur_frame_ ++;

#ifdef LOG_TIME
    hrt_.stop();
    cout << "time spent for update state (possibly finetune): " << hrt_.getMilliseconds() << " ms" << endl;
#endif
}