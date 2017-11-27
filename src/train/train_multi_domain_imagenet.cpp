// Train the neural network tracker.

#include <string>
#include <iostream>

#include <caffe/caffe.hpp>

#include "example_generator.h"
#include "helper/helper.h"
#include "loader/loader_imagenet_det.h"
#include "loader/loader_otb.h"
#include "network/regressor_train.h"
#include "train/tracker_trainer_multi_domain.h"
#include "tracker/tracker_manager.h"
#include "loader/video_imagenet.h"
#include "loader/loader_imagenet_video.h"

using std::string;

// Desired number of training batches.
const int kNumBatches = 500000;

// Desired number of iterations, within each iteration, each of K domains is passed 1 batch
// const int NUM_CYCLES = 800;
const int NUM_CYCLES = 3200;

namespace {

// Train on all annotated frames in the set of videos.
void train_video(const std::vector<VideoImageNet>& videos, TrackerTrainerMultiDomain* tracker_trainer_multi_domain) {
  // Get a random video.
  const int video_num = rand() % videos.size();
  const VideoImageNet& video = videos[video_num];
  const vector<string> & frames = video.all_frames;

  // We need at least 2 annotations in this video for this to be useful.
  if (!video.hasEnoughAnnotation()) {
    printf("Error - video %s has not have enough annotations\n", video.path.c_str());
    return;
  }

  // Choose a random annotation.
  int frame_index = rand() % (frames.size() - 1);
  vector<int> common_trackids;
  video.CheckTwoFramesCommonTrackObject(frame_index, frame_index + 1, common_trackids);
  while (common_trackids.size() == 0) {
    frame_index = rand() % (frames.size() - 1); // sample another frame
    video.CheckTwoFramesCommonTrackObject(frame_index, frame_index + 1, common_trackids);
  }
  
  // Choose a random track object
  int track_object_id = common_trackids[rand() % (common_trackids.size())];
  
  // Load the frame's annotation.
  int frame_num_prev;
  cv::Mat image_prev;
  BoundingBox bbox_prev;
  bool prev_success = video.LoadFrame(frame_index, track_object_id, false, false, &image_prev, &bbox_prev);

  // Load the next frame's annotation.
  int frame_num_curr;
  cv::Mat image_curr;
  BoundingBox bbox_curr;
  bool curr_success = video.LoadFrame(frame_index + 1, track_object_id, false, false, &image_curr, &bbox_curr);

  // make sure both bboxes are valid before process
  while((!prev_success) || (!curr_success) || (!(bbox_prev.valid_bbox() && bbox_curr.valid_bbox()))) {
    // repick the frame_index
    frame_index = rand() % (frames.size() - 1);
    
    // recheck common objects
    common_trackids.clear();
    video.CheckTwoFramesCommonTrackObject(frame_index, frame_index + 1, common_trackids);
    while (common_trackids.size() == 0) {
        frame_index = rand() % (frames.size() - 1); // sample another frame
        video.CheckTwoFramesCommonTrackObject(frame_index, frame_index + 1, common_trackids);
    }
    
    // Choose a random track object
    track_object_id = common_trackids[rand() % (common_trackids.size())];
    
    // Load the frame's annotation.
    prev_success = video.LoadFrame(frame_index, track_object_id, false, false, &image_prev, &bbox_prev);

    // Load the next frame's annotation.
    curr_success = video.LoadFrame(frame_index + 1, track_object_id, false, false, &image_curr, &bbox_curr);
  }

  // Train on this example, actually enqueue this example, if batch filled, train
  tracker_trainer_multi_domain->Train(image_prev, image_curr, bbox_prev, bbox_curr);

  // Save
  frame_num_prev = frame_num_curr;
  image_prev = image_curr;
  bbox_prev = bbox_curr;
}

} // namespace

int main (int argc, char *argv[]) {
  if (argc < 13) {
    std::cerr << "Usage: " << argv[0]
              << " imagenet_video_data_folder"
              << " imagenet_video_annotation_folder"
              << " network.caffemodel train.prototxt"
              << " mean_file"
              << " solver_file"
              << " lambda_shift lambda_scale min_scale max_scale"
              << " gpu_id"
              << " random_seed"
              << std::endl;
    return 1;
  }

  FLAGS_alsologtostderr = 1;

  ::google::InitGoogleLogging(argv[0]);

  int arg_index = 1;
  const string& imagenet_video_data_folder        = argv[arg_index++];
  const string& imagenet_video_annotation_folder  = argv[arg_index++];
  const string& caffe_model   = argv[arg_index++];
  const string& train_proto   = argv[arg_index++];
  const string& mean_file     = argv[arg_index++];
  const string& solver_file  = argv[arg_index++];
  const double lambda_shift        = atof(argv[arg_index++]);
  const double lambda_scale        = atof(argv[arg_index++]);
  const double min_scale           = atof(argv[arg_index++]);
  const double max_scale           = atof(argv[arg_index++]);
  const int gpu_id          = atoi(argv[arg_index++]);
  const int random_seed          = atoi(argv[arg_index++]);

  caffe::Caffe::set_random_seed(random_seed);
  printf("Using random seed: %d\n", random_seed);

#ifdef CPU_ONLY
  printf("Setting up Caffe in CPU mode\n");
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
  printf("Setting up Caffe in GPU mode with ID: %d\n", gpu_id);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::SetDevice(gpu_id);
#endif


  // Load the video data.
  LoaderImageNetVideo imagenet_video_loader(imagenet_video_data_folder, imagenet_video_annotation_folder);
  std::vector<VideoImageNet> train_videos = imagenet_video_loader.get_videos();
  printf("Total training videos: %zu\n", train_videos.size());

  int K = -1; // single domain training

  // Create an ExampleGenerator to generate training examples.
  ExampleGenerator example_generator(lambda_shift, lambda_scale,
                                     min_scale, max_scale);

  // save the loss_history when done, TODO: save loss along training instead end of training
  string save_dir = "loss_history/";
  string save_path = save_dir + "train_single_domain_loss_no_middle_batch_no_pool_avg_history_cycle" + std::to_string(NUM_CYCLES) + ".txt";
  if (!boost::filesystem::exists(save_dir)) {
    boost::filesystem::create_directories(save_dir);
  }
  if (boost::filesystem::exists(save_path)) {
    // clean previous run loss log
    boost::filesystem::remove(save_path);
  }

  // Set up network.
  RegressorTrain regressor_train(train_proto, caffe_model, mean_file
                                 gpu_id, solver_file, save_path, K);

  // Set up trainer.
  TrackerTrainerMultiDomain tracker_trainer_multi_domain(&example_generator, &regressor_train);

  for (int i = 0;i < kNumBatches; i ++) {
    train_video(train_videos, &tracker_trainer_multi_domain);
  }

  return 0;
}