#ifndef CONSTANTS_H
#define CONSTANTS_H

#define EPSILON (1e-03)

#define TOP_ESTIMATES 5 // number of top candidates to do the estimates, heuristic

// for tracker motion model sample candidates
#define SAMPLE_CANDIDATES 250

// for gaussian sampling
#define SD_X 0.3 // translation std: mean(width,height)*SD_X
#define SD_Y 0.3
#define SD_SCALE 0.5 
#define KEEP_SD 2.0
#define MOTION_SCALE_FACTOR 1.05 // scaling std: MOTION_SCALE_FACTOR^(SD_SCALE)

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

// for training labels
#define POS_LABEL 1.0
#define NEG_LABEL 0.0

#endif