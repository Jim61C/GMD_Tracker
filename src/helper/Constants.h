#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "CommonCV.h"

#define EPSILON (1e-03)
#define PI 3.1415926

#define TOP_ESTIMATES 5 // number of top candidates to do the estimates, heuristic

// for tracker motion model sample candidates
#define SAMPLE_CANDIDATES 250

// for gaussian sampling
#define SD_X 0.3 // translation std: mean(width,height)*SD_X
#define SD_Y 0.3
#define SD_SCALE 1.0 
#define KEEP_SD 2.0
#define MOTION_SCALE_FACTOR 1.05 // scaling std: MOTION_SCALE_FACTOR^(SD_SCALE)

#define SD_AP 1.0
#define MOTION_AP_FACTOR 1.05

// for random sampling
#define SCALE_FACTOR 1.05
#define POS_SCALE_RANGE 5.0 // for positive sample 
#define POS_TRANS_RANGE 0.1 // for positive sample

#define NEG_SCALE_RANGE 10 // for negative sample
#define NEG_TRANS_RANGE 2 // for negative sample 

// for IOU training sample generation
const int POS_CANDIDATES = 50;
const int NEG_CANDIDATES = 200;
const double POS_IOU_TH = 0.7;
const double NEG_IOU_TH = 0.5;

// for inner mini batch in training
const int INNER_BATCH_SIZE = 50;

// for fine tune sample generation
const int POS_CANDIDATES_FINETUNE = 10;
const int NEG_CANDIDATES_FINETUNE = 40;
// const int NOHEM_FINETUNE = 40;

// for training labels
#define POS_LABEL 1.0
#define NEG_LABEL 0.0

// long/short term update
#define LONG_TERM_BAG_SIZE 50
#define SHORT_TERM_BAG_SIZE 20
#define LONG_TERM_UPDATE_INTERVAL 20
#define LONG_TERM_POS_CANDIDATE_UPPER_BOUND 10
#define LONG_TERM_NEG_CANDIDATE_UPPER_BOUND 200 // number of examples for forwarding, backward only does NOHEM_FINETUNE number of negative sampels
const double SHORT_TERM_FINE_TUNE_TH = 0.5; // if want less frequent short term fine tune when distance window is applied, make if < 0.5
#define ADD_DISTANCE_PENALTY

// DEBUGGING
#define SEED_RNG_EXAMPLE_GENERATOR 800
#define SEED_RNG_TRACKER 500
#define SEED_ENGINE 800

// Online Learning
// #define FIRST_FRAME_FINETUNE_ITERATION 10
// #define FIRST_FRAME_POS_SAMPLES 50
// #define FIRST_FRAME_NEG_SAMPLES 500

#define FIRST_FRAME_FINETUNE_ITERATION 30
#define FIRST_FRAME_POS_SAMPLES 32
#define FIRST_FRAME_NEG_SAMPLES 1000

#define FIRST_FRAME_NUM_MINI_BATCH 2

#define ONHEM_BASE 96
#define FIRST_FRAME_ONHEM ONHEM_BASE/FIRST_FRAME_NUM_MINI_BATCH // online hard examples used


// ROI Pooling
const double TARGET_SIZE = 600.0; // compare to min (W, H)
const double MAX_SIZE = 1000.0; // make sure the image_curr does not exceed this size

// network input index
#define TARGET_NETWORK_INPUT_IDX 0
#define CANDIDATE_NETWORK_INPUT_IDX 1
#define ROIS_NETWORK_INPUT_IDX 2
#define LABEL_NETWORK_INPUT_IDX 3

// training image mean 
const cv::Scalar mean_scalar(104, 117, 123);

// distance penalty padding
#define DISTANCE_PENALTY_PAD 0

// for BoundingBox Regression
const int BBOX_REGRESSION_FEATURE_LENGTH = 6 * 6 * 256;

#endif