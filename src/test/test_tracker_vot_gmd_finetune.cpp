// This file was mostly taken from the example given here:
// http://www.votchallenge.net/howto/integration.html

// Uncomment line below if you want to use rectangles
#define VOT_RECTANGLE
#include "native/vot.h"

#include <string>

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

const bool show_intermediate_output = false;

int main (int argc, char *argv[]) {
   if (argc < 8) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel solver_file LAMBDA_SHIFT LAMBDA_SCALE MIN_SCALE MAX_SCALE"
              << " [gpu_id]" << std::endl;
    return 1;
  }

  // FLAGS_alsologtostderr = 1;

  ::google::InitGoogleLogging(argv[0]);

  const string& model_file   = argv[1];
  const string& trained_file = argv[2];
  const string& solver_file = argv[3];
  const double lambda_shift   = atof(argv[4]);
  const double lambda_scale   = atof(argv[5]);
  const double min_scale      = atof(argv[6]);
  const double max_scale      = atof(argv[7]);

  int gpu_id = 0;
  if (argc >= 9) {
    gpu_id = atoi(argv[8]);
  }

  // Set up the neural network.
  const bool do_train = true;
  RegressorTrain regressor_train(model_file,
                               trained_file,
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

  VOT vot; // Initialize the communcation

  // Get region and first frame
  VOTRegion region = vot.region();
  string path = vot.frame();
  string next_path = vot.frame();

  // Load the first frame and use the initialization region to initialize the tracker, since finetune tracker, also does first frame fine tune
  tracker_gmd.Init(path, region, &regressor_train);

  //track
  while (true) {
      path = next_path;
      if (path.empty()) break; // Are we done?
      next_path = vot.frame();

      // Load current image.
      const cv::Mat& image = cv::imread(path);

      // Track and estimate the bounding box location.
      BoundingBox bbox_estimate;
      tracker_gmd.Track(image, &regressor_train, &bbox_estimate);

      // After estimation, update state, previous frame, new bbox priors etc, if no next_path, then current path is the last one
      tracker_gmd.UpdateState(image, bbox_estimate, &regressor_train, next_path.empty());

      bbox_estimate.GetRegion(&region);

      vot.report(region); // Report the position of tracker_gmd
  }

  // Finishing the communication is completed automatically with the destruction
  // of the communication object (if you are using pointers you have to explicitly
  // delete the object).

  return 0;
}
