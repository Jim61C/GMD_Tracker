GPU_ID=0
FOLDER=GOTURN_MDNET
RANDOM_SEED=800

SOLVER=nets/solver.prototxt
TRAIN_PROTO=nets/tracker_new_train.prototxt
CAFFE_MODEL=nets/models/weights_init/tracker_init.caffemodel

BASEDIR=nets
RESULT_DIR=$BASEDIR/results/$FOLDER
SOLVERSTATE_DIR=$BASEDIR/solverstate/$FOLDER

#Make folders to store results and snapshots
mkdir -p $RESULT_DIR
mkdir -p $SOLVERSTATE_DIR

#Modify solver to save snapshot in SOLVERSTATE_DIR
mkdir -p nets/solver_temp
SOLVER_TEMP=nets/solver_temp/solver_temp_$FOLDER.prototxt
sed s#SOLVERSTATE_DIR#$SOLVERSTATE_DIR# <$SOLVER >$SOLVER_TEMP
sed -i s#TRAIN_FILE#$TRAIN_PROTO# $SOLVER_TEMP
sed -i s#DEVICE_ID#$GPU_ID# $SOLVER_TEMP
sed -i s#RANDOM_SEED#$RANDOM_SEED# $SOLVER_TEMP
