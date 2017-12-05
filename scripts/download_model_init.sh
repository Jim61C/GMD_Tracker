#!/bin/bash

mkdir -p nets/models/weights_init 
cd nets/models/weights_init
wget http://cs.stanford.edu/people/davheld/public/GOTURN/weights_init/tracker_init.caffemodel
cd ../../../


cd nets/models/weights_init
wget http://www.jimxingyf.com/static/models/VGG_CNN_M.caffemodel
cd ../../../
