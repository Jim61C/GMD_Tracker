GOTURN_MDNET-Tracker Dependencies information
======================
OS: Ubuntu16.04

Language: C++

Compiler: g++-5

Flags: C++11

Dependencies:
- CUDA8.0
- OpenCV3.2.0
- caffe-fast-rcnn
- Boost1.x.x
- Eigen3.x.x
- protobuf3.x.x
- GSL
- GLOG
- TRAX

Hardware:
- GPU: NVIDIA GPU Compute capability >= 3.7 and minimum 3 GB graphics memory needed
- Driver: NVIDIA Graphics Driver 375

Instructions for Compilation
======================
Note that the binaries 'trax_tracker_no_middle_batch_single_no_pool_avg' and 'libGOTURN.a' are given for the sake of direct use without compilation but after install the dependencies and compilation, the same binaries will be built.

Running ./install.sh should install the necessary dependencies, please make sure you have sudo access to the machine as some of the dependencies will be built in /usr/local/include, /usr/include and /usr/lib/x86_64-linux-gnu.
For troubleshooting, please refer to the comments in install.sh

Questions
======================
For further questions, please contact <yxing1@andrew.cmu.edu>
