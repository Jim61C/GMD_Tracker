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
#include "loader/video.h"
#include "loader/video_loader.h"

using std::string;

// Desired number of iterations, within each iteration, each of K domains is passed 1 batch
const int NUM_CYCLES = 200;

namespace {

// Train on a random image.
// void train_image(const LoaderImagenetDet& image_loader,
//            const std::vector<std::vector<Annotation> >& images,
//            TrackerTrainer* tracker_trainer) {
//   // Get a random image.
//   const int image_num = rand() % images.size();
//   const std::vector<Annotation>& annotations = images[image_num];

//   // Choose a random annotation.
//   const int annotation_num = rand() % annotations.size();

//   // Load the image with its ground-truth bounding box.
//   cv::Mat image;
//   BoundingBox bbox;
//   image_loader.LoadAnnotation(image_num, annotation_num, &image, &bbox);

//   // Train on this example
//   tracker_trainer->Train(image, image, bbox, bbox);
// }

void train_video_k(const Video &video, TrackerTrainerMultiDomain* tracker_trainer_multi_domain) {
  // get one random frame from video k and invoke tracker_trainer_multi_domain.Train()
  
  // Get the video's annotations.
  const std::vector<Frame>& annotations = video.annotations;

  // We need at least 2 annotations in this video for this to be useful.
  if (annotations.size() < 2) {
    printf("Error - video %s has only %zu annotations\n", video.path.c_str(),
           annotations.size());
    return;
  }

  // Choose a random annotation.
  int annotation_index = rand() % (annotations.size() - 1);

  // Load the frame's annotation.
  int frame_num_prev;
  cv::Mat image_prev;
  BoundingBox bbox_prev;
  video.LoadAnnotation(annotation_index, &frame_num_prev, &image_prev, &bbox_prev);

  // Load the next frame's annotation.
  int frame_num_curr;
  cv::Mat image_curr;
  BoundingBox bbox_curr;
  video.LoadAnnotation(annotation_index + 1, &frame_num_curr, &image_curr, &bbox_curr);

  // make sure both bboxes are valid before process
  while(!(bbox_prev.valid_bbox() && bbox_curr.valid_bbox())) {
    // reload
    annotation_index = rand() % (annotations.size() - 1);
    video.LoadAnnotation(annotation_index, &frame_num_prev, &image_prev, &bbox_prev);
    video.LoadAnnotation(annotation_index + 1, &frame_num_curr, &image_curr, &bbox_curr);
  }

  // Train on this example, actually enqueue into batch
  tracker_trainer_multi_domain->Train(image_prev, image_curr, bbox_prev, bbox_curr);

  // Save
  frame_num_prev = frame_num_curr;
  image_prev = image_curr;
  bbox_prev = bbox_curr;
  return;
}

// Train one batch given a domain index
void train_video_k_one_batch(const std::vector<Video>& videos, TrackerTrainerMultiDomain* tracker_trainer_multi_domain, int k) {
  // fill until one batch is full for video k
  const Video& video = videos[k];
  tracker_trainer_multi_domain->set_current_k(k);

  while (!tracker_trainer_multi_domain->get_if_full_batch()) {
    // generate and enqueue examples to batch
    cout << "generate and enqueue one frame examples for domain " << k << endl;
    train_video_k(video, tracker_trainer_multi_domain);
  }
  
  // set batch_filled_ as false again for next batch
  tracker_trainer_multi_domain->set_batch_filled(false);
  // remove remaining batches as k might be different for next batch
  tracker_trainer_multi_domain->clear_batch_remaining();

  return;
}

// Train on all annotated frames in the set of videos.
void train_video(const std::vector<Video>& videos, TrackerTrainerMultiDomain* tracker_trainer_multi_domain) {
  // Get a random video.
  const int video_num = rand() % videos.size();
  const Video& video = videos[video_num];

  // Get the video's annotations.
  const std::vector<Frame>& annotations = video.annotations;

  // We need at least 2 annotations in this video for this to be useful.
  if (annotations.size() < 2) {
    printf("Error - video %s has only %zu annotations\n", video.path.c_str(),
           annotations.size());
    return;
  }

  // Choose a random annotation.
  int annotation_index = rand() % (annotations.size() - 1);

  // Load the frame's annotation.
  int frame_num_prev;
  cv::Mat image_prev;
  BoundingBox bbox_prev;
  video.LoadAnnotation(annotation_index, &frame_num_prev, &image_prev, &bbox_prev);

  // Load the next frame's annotation.
  int frame_num_curr;
  cv::Mat image_curr;
  BoundingBox bbox_curr;
  video.LoadAnnotation(annotation_index + 1, &frame_num_curr, &image_curr, &bbox_curr);

  // make sure both bboxes are valid before process
  while(!(bbox_prev.valid_bbox() && bbox_curr.valid_bbox())) {
    // reload
    annotation_index = rand() % (annotations.size() - 1);
    video.LoadAnnotation(annotation_index, &frame_num_prev, &image_prev, &bbox_prev);
    video.LoadAnnotation(annotation_index + 1, &frame_num_curr, &image_curr, &bbox_curr);
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
  if (argc < 11) {
    std::cerr << "Usage: " << argv[0]
              << " otb_videos_folder"
              << " network.caffemodel train.prototxt"
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
  const string& otb_videos_folder      = argv[arg_index++];
  const string& caffe_model   = argv[arg_index++];
  const string& train_proto   = argv[arg_index++];
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
  LoaderOTB otb_video_loader(otb_videos_folder);
  std::vector<Video> train_videos = otb_video_loader.get_videos();
  printf("Total training videos: %zu\n", train_videos.size());

  int K = train_videos.size(); // multi domain training

  // Create an ExampleGenerator to generate training examples.
  ExampleGenerator example_generator(lambda_shift, lambda_scale,
                                     min_scale, max_scale);

  // save the loss_history when done, TODO: save loss along training instead end of training
  string save_dir = "loss_history/";
  string save_path = save_dir + "train_mdnet_setdiff_" + std::to_string(NUM_CYCLES) + ".txt";
  if (!boost::filesystem::exists(save_dir)) {
    boost::filesystem::create_directories(save_dir);
  }
  if (boost::filesystem::exists(save_path)) {
    // clean previous run loss log
    boost::filesystem::remove(save_path);
  }

  // Set up network.
  RegressorTrain regressor_train(train_proto, caffe_model,
                                 gpu_id, solver_file, save_path, K);

  // Set up trainer.
  TrackerTrainerMultiDomain tracker_trainer_multi_domain(&example_generator, &regressor_train);

  for (int i = 0; i < NUM_CYCLES; i++) {
    cout << "CYCLE " << i << endl;
    for (int k = 0; k < K; k ++) {
      cout << "Domain " << k << endl;
      train_video_k_one_batch(train_videos, &tracker_trainer_multi_domain, k);
    }
  }

  return 0;
}