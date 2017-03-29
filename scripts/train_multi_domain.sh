VIDEOS_FOLDER=~/Downloads/Tracking_Sequences/OTB-VOT2014/
CAFFE_MODEL=nets/models/weights_init/tracker_init.caffemodel
TRAIN_PROTO=nets/tracker_new_train.prototxt
SOLVER=nets/solver_temp/solver_temp_GOTURN_MDNET.prototxt
LAMBDA_SHIFT=5
LAMBDA_SCALE=15
MIN_SCALE=-0.4
MAX_SCALE=0.4
GPU_ID=0
RANDOM_SEED=800

./build/train_multi_domain $VIDEOS_FOLDER $CAFFE_MODEL $TRAIN_PROTO $SOLVER $LAMBDA_SHIFT $LAMBDA_SCALE $MIN_SCALE $MAX_SCALE $GPU_ID $RANDOM_SEED