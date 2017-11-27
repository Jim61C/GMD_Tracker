#include <string>
#include <caffe/caffe.hpp>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "network/regressor.h"
#include "loader/loader_vot.h"
#include "tracker/tracker.h"
#include "tracker/tracker_gmd.h"
#include "tracker/tracker_manager.h"

// for fine tuning
#include "network/regressor_train.h"
#include "train/example_generator.h"
#include "train/tracker_trainer_multi_domain.h"

// trax protocol
#include <trax.h>
#include <trax/opencv.hpp>

const bool show_intermediate_output = false;

int main (int argc, char *argv[]) {
   if (argc < 9) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel mean_file solver_file LAMBDA_SHIFT LAMBDA_SCALE MIN_SCALE MAX_SCALE"
              << " [gpu_id]" << std::endl;
    return 1;
  }

  // FLAGS_alsologtostderr = 1;

  ::google::InitGoogleLogging(argv[0]);
//   caffe::Caffe::set_random_seed(800); 
  int arg_idx = 1;
  const string& model_file   = argv[arg_idx++];
  const string& trained_file = argv[arg_idx++];
  const string& mean_file = argv[arg_idx++];
  const string& solver_file = argv[arg_idx++];
  const double lambda_shift   = atof(argv[arg_idx++]);
  const double lambda_scale   = atof(argv[arg_idx++]);
  const double min_scale      = atof(argv[arg_idx++]);
  const double max_scale      = atof(argv[arg_idx++]);

  int gpu_id = 0;
  if (argc > arg_idx ) {
    gpu_id = atoi(argv[arg_idx++]);
  }

  // Set up the neural network.
  const bool do_train = true;
  RegressorTrain regressor_train(model_file,
                               trained_file,
                               mean_file,
                               gpu_id,
                               solver_file,
                               4,
                               do_train);

  // Get example_generator
  ExampleGenerator example_generator(lambda_shift, lambda_scale,
                                    min_scale, max_scale);
  
  // Create a tracker object.
  TrackerGMD tracker_gmd(show_intermediate_output, &example_generator, &regressor_train);

  // Ensuring randomness for fairness.
  srandom(time(NULL));
  
  int run = 1;

  trax::Server handle(trax::Metadata(TRAX_REGION_RECTANGLE,
                                TRAX_IMAGE_PATH), trax_no_log);

  BoundingBox bbox_gt;

  while (run) {
      trax::Image image;
      trax::Region region;
      trax::Properties properties;
      int tr = handle.wait(image, region, properties);
      if (tr == TRAX_INITIALIZE) {
          // init tracker_gmd
          cv::Mat image_curr = trax::image_to_mat(image);
          cv::Mat image_track = image_curr.clone();
          cv::Rect bbox_rect = trax::region_to_rect(region);
          bbox_gt = BoundingBox(bbox_rect.x, bbox_rect.y, bbox_rect.x + bbox_rect.width, bbox_rect.y+ bbox_rect.height);
          tracker_gmd.Init(image_track, bbox_gt,  &regressor_train);

          cv::Rect result(bbox_gt.x1_, bbox_gt.y1_, bbox_gt.get_width(), bbox_gt.get_height());
          handle.reply(trax::rect_to_region(result), trax::Properties());

      } else if (tr == TRAX_FRAME) {
          cv::Mat image_curr = trax::image_to_mat(image);
          cv::Mat image_track = image_curr.clone();
          // Track and estimate the bounding box location.
          BoundingBox bbox_estimate;
          tracker_gmd.Track(image_track, &regressor_train, &bbox_estimate);

          // After estimation, update state; Here assume no last frame, TODO: try read in next_tr and parse, see if trax protocol still works
          tracker_gmd.UpdateState(image_track, bbox_estimate, &regressor_train, false);
            
          // report result
          cv::Rect result(bbox_estimate.x1_, bbox_estimate.y1_, bbox_estimate.get_width(), bbox_estimate.get_height());
          handle.reply(trax::rect_to_region(result), trax::Properties());
      }
      else {
          run = 0;
      };
  }

  return 0;
}
