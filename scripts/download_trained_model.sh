#!/bin/bash

mkdir -p nets/solverstate/GOTURN_MDNET
cd nets/solverstate/GOTURN_MDNET
wget http://www.jimxingyf.com/static/models/caffenet_train_GOTURN_MDNET_IMAGENET.caffemodel.h5
cd ../../../


mkdir -p nets/solverstate/MDNET
cd nets/solverstate/MDNET
wget http://www.jimxingyf.com/static/models/caffenet_train_MDNET_OTB-VOT2014.caffemodel.h5
wget http://www.jimxingyf.com/static/models/caffenet_train_MDNET_OTB-VOT2015.caffemodel
wget http://www.jimxingyf.com/static/models/caffenet_train_MDNET_OTB-VOT2016.caffemodel
cd ../../../


mkdir -p nets/solverstate/MDNET_ROIPOOL
cd nets/solverstate/MDNET_ROIPOOL
wget http://www.jimxingyf.com/static/models/caffenet_train_MDNET_ROIPOOL_OTB-VOT2014.caffemodel.h5
cd ../../../
