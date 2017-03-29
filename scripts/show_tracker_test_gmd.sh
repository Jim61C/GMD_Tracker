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
DEPLOY=nets/tracker_new_test.prototxt
CAFFE_MODEL=nets/solverstate/GOTURN_MDNET/caffenet_train_iter_150000.caffemodel

build/show_tracker_vot_gmd $DEPLOY $CAFFE_MODEL $VIDEOS_FOLDER $GPU_ID $START_VIDEO_NUM $PAUSE_VAL
