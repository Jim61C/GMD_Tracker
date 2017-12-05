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
Firstly, clone the respository to folder 'volvo_tracker':
```
git clone https://github.com/Jim61C/Volvo_Capstone_Tracker.git volvo_tracker
```
Then, copy out the installation script to the same directory level as volvo_tracker
```
cp ./volvo_tracker/install.sh .
./install.sh
```
Running ./install.sh should install the necessary dependencies, please make sure you have sudo access to the machine as some of the dependencies will be built in /usr/local/include, /usr/include and /usr/lib/x86_64-linux-gnu.
For troubleshooting, please refer to the comments in install.sh

Instructions for running different models
======================
Note that the source for different models is associated with different tags, checkout to the specific tag to run the specific tracker model. The trained models can be downloaded with
```
bash ./scripts/download_trained_model.sh
```
The four models are summarized in the following:
- Model1.0: MDNet C++ implementation, tag: model1.0
- Model1.1: MDNet C++ implementation with custom online update scheme, tag: model1.1
- Model2.0: MDNet C++ with ROI Pooling, tag: model2.0
- Model3.0: Siamese CNN racker (GOTURN + MDNET), tag: model3.0

For example, to run Model1.0 against VOT2014 Benchmark, check out to tag model1.0 and then:
```
./scripts/show_mdnet.sh <path to vot2014 dataset folder>
```

To run Model1.1 against VOT2014 Benchmark, check out to tag model1.1 and then:
```
./scripts/show_mdnet.sh <path to vot2014 dataset folder>
```

To run Model2.0 against VOT2014 Benchmark, check out to tag model2.0 and then:
```
./scripts/show_mdnet_roipool.sh <path to vot2014 dataset folder>
```

To run Model3.0 against VOT2014 Benchmark, check out to tag model3.0 and then:
```
./scripts/show_tracker_single_domain_two_stream_roi_batch.sh <path to vot2014 dataset folder>
```

Instructions for training the models
======================
Firstly, initial CaffeNet and VGG-M net models in caffe needs to be downloaded.
```
bash ./scripts/download_model_init.sh
```
Then, check out to the specific tag and then run the training script for that particular model.
For example, to train the model3.0, checkout to tag model3.0, then
```
./scripts/train_single_domain_two_stream_rois_batch.sh <IMAGENET_VIDEO_DETECTION_VIDEO_FOLDER> <IMAGENET_VIDEO_DETECTION_ANNOTATION_FOLDER>
```
To train the model1.0/1.1, checkout to right tags, then
```
./scripts/train_mdnet.sh <path to OTB-VOT2014/15/16 folder>
```
To train the model2.0, checkout to right tag, then
```
./scripts/train_mdnet_roipool.sh <path to OTB-VOT2014/15/16 folder>
```


Performance Comparison
======================
The following table shows the performance and speed summary of the different models.

| Metric          | Model1.0      | Model1.1 | Model2.0  | Model3.0 | DSST | MDNet Matlab | GOTURN |
| :-------------: |:-------------:| :-----:  |:--------: |:--------:| :---:| :-----------:| :-----:|
| Expected Overlap| 0.4404        | 0.4178   |0.3516     | 0.2909   |0.2810| 0.4534       |0.2409  |
| FPS             |   1.5         | 1.5      | 17.2      |  15.6    | 25.4 | 1.0          |> 100   |


The shows Model1.0 on VOT2014 Benchmark

The shows Model1.1 on VOT2014 Benchmark

The shows Model1.1 on VOT2016 Benchmark

The shows Model2.0 on VOT2014 Benchmark

The shows Model3.0 on VOT2014 Benchmark

The following is the video result on VOT2014 Benchmark (Model1.0, MDNet C++)
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/IRY7MwfJIR0/0.jpg)](https://www.youtube.com/watch?v=IRY7MwfJIR0)

The following is the vidoe result on VOT2016 Benchmark (Model1.1, MDNet C++ Custom)
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/eCof5qNE3eU/0.jpg)](https://www.youtube.com/watch?v=eCof5qNE3eU)

Questions
======================
For further questions, please contact <yxing1@andrew.cmu.edu>
