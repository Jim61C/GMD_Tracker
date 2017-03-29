#!/bin/bash

if [ -z "$1" ] 
  then
    echo "No folder supplied!"
    echo "Usage: bash `basename "$0"` vot_video_folder"
    exit
fi

# Choose which GPU the tracker runs on
GPU_ID=0

# Choose which video from the test set to start displaying
START_VIDEO_NUM=0

# Set to 0 to pause after each frame
PAUSE_VAL=1

VIDEOS_FOLDER=$1
DEPLOY=nets/tracker_new_finetune.prototxt
CAFFE_MODEL=nets/solverstate/GOTURN_MDNET/caffenet_train_iter_150000.caffemodel
SOLVER_FILE=nets/solver_temp/solver_temp_GOTURN_MDNET_FINETUNE.prototxt
LAMBDA_SHIFT=5
LAMBDA_SCALE=15
MIN_SCALE=-0.4
MAX_SCALE=0.4

build/show_tracker_vot_gmd_finetune $DEPLOY $CAFFE_MODEL $SOLVER_FILE $VIDEOS_FOLDER $LAMBDA_SHIFT $LAMBDA_SCALE $MIN_SCALE $MAX_SCALE $GPU_ID $START_VIDEO_NUM $PAUSE_VAL
