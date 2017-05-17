if [ -z "$1" ] 
  then
    echo "No folder supplied!"
    echo "Usage: bash `basename "$0"` vot_video_folder"
    exit
fi
VIDEOS_FOLDER=$1
CAFFE_MODEL=nets/models/weights_init/tracker_init.caffemodel
TRAIN_PROTO=nets/tracker_new_train.prototxt
SOLVER=nets/solver_temp/solver_temp_GOTURN_MDNET_TWO_STREAM_ROIS_BATCH_SINGLE_NO_POOL_AVG.prototxt
LAMBDA_SHIFT=5
LAMBDA_SCALE=15
MIN_SCALE=-0.4
MAX_SCALE=0.4
GPU_ID=0
RANDOM_SEED=800

./build/train_single_domain_no_middle_batch_no_pool_avg $VIDEOS_FOLDER $CAFFE_MODEL $TRAIN_PROTO $SOLVER $LAMBDA_SHIFT $LAMBDA_SCALE $MIN_SCALE $MAX_SCALE $GPU_ID $RANDOM_SEED
