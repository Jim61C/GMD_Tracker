#include "tracker_manager.h"

#include <string>

#include "helper/helper.h"
#include "train/tracker_trainer.h"

using std::string;

#define FINE_TUNE_AUGMENT_NUM 10
#define FISRT_FRAME_PAUSE

TrackerManager::TrackerManager(const std::vector<Video>& videos,
                               RegressorBase* regressor, Tracker* tracker) :
  videos_(videos),
  regressor_(regressor),
  tracker_(tracker)
{
}

void TrackerManager::TrackAll() {
  TrackAll(0, 1);
}

void TrackerManager::TrackAll(const size_t start_video_num, const int pause_val) {
  // Iterate over all videos and track the target object in each.
  for (size_t video_num = start_video_num; video_num < videos_.size(); ++video_num) {
    // Get the video.
    const Video& video = videos_[video_num];

    // Perform any pre-processing steps on this video.
    VideoInit(video, video_num);

    // Get the first frame of this video with the initial ground-truth bounding box (to initialize the tracker).
    int first_frame;
    cv::Mat image_curr;
    BoundingBox bbox_gt;
    video.LoadFirstAnnotation(&first_frame, &image_curr, &bbox_gt);

    // Initialize the tracker.
    tracker_->Init(image_curr, bbox_gt, regressor_);

    // Iterate over the remaining frames of the video.
    for (size_t frame_num = first_frame + 1; frame_num < video.all_frames.size(); ++frame_num) {

      // Get image for the current frame.
      // (The ground-truth bounding box is used only for visualization).
      const bool draw_bounding_box = false;
      const bool load_only_annotation = false;
      cv::Mat image_curr;
      BoundingBox bbox_gt;
      bool has_annotation = video.LoadFrame(frame_num,
                                            draw_bounding_box,
                                            load_only_annotation,
                                            &image_curr, &bbox_gt);

      // Get ready to track the object.
      SetupEstimate();

      // Track and estimate the target's bounding box location in the current image.
      // Important: this method cannot receive bbox_gt (the ground-truth bounding box) as an input.
      BoundingBox bbox_estimate_uncentered;
      tracker_->Track(image_curr, regressor_, &bbox_estimate_uncentered);

      // Process the output (e.g. visualize / save results).
      ProcessTrackOutput(frame_num, image_curr, has_annotation, bbox_gt,
                           bbox_estimate_uncentered, pause_val);
    }
    PostProcessVideo();
  }
  PostProcessAll();
}

TrackerVisualizer::TrackerVisualizer(const std::vector<Video>& videos,
                                     RegressorBase* regressor, Tracker* tracker) :
  TrackerManager(videos, regressor, tracker)
{
}


void TrackerVisualizer::ProcessTrackOutput(
    const size_t frame_num, const cv::Mat& image_curr, const bool has_annotation,
    const BoundingBox& bbox_gt, const BoundingBox& bbox_estimate_uncentered,
    const int pause_val) {
  cv::Mat full_output;
  image_curr.copyTo(full_output);

  if (has_annotation) {
    // Draw ground-truth bounding box of the target location (white).
    bbox_gt.DrawBoundingBox(&full_output);
  }

  // Draw estimated bounding box of the target location (red).
  bbox_estimate_uncentered.Draw(255, 0, 0, &full_output);

  // Show the image with the estimated and ground-truth bounding boxes.
  cv::namedWindow("Full output", cv::WINDOW_AUTOSIZE ); // Create a window for display.
  cv::imshow("Full output", full_output );                   // Show our image inside it.

  // Pause for pause_val milliseconds, or until user input (if pause_val == 0).
  cv::waitKey(pause_val);
}

void TrackerVisualizer::VideoInit(const Video& video, const size_t video_num) {
  printf("Video: %zu\n", video_num);
}


// Tracker With Fine Tuning Ability
TrackerFineTune::TrackerFineTune(const std::vector<Video>& videos,
                                RegressorBase* regressor, Tracker* tracker,
                                ExampleGenerator* example_generator,
                                RegressorTrainBase* regressor_train) :
  TrackerManager(videos, regressor, tracker),
  example_generator_(example_generator),
  regressor_train_(regressor_train)

{

}


void TrackerFineTune::VideoInit(const Video& video, const size_t video_num) {
  printf("In TrackerFineTune, Video: %zu\n", video_num);

  printf("About to fine tune the first frame");
  
  // Get the first frame of this video with the initial ground-truth bounding box (to initialize the tracker).
  int first_frame;
  cv::Mat image_curr;
  BoundingBox bbox_gt;
  video.LoadFirstAnnotation(&first_frame, &image_curr, &bbox_gt);

#ifdef FISRT_FRAME_PAUSE
  cv::Mat image_curr_show = image_curr.clone();
  bbox_gt.DrawBoundingBox(&image_curr_show);
  cv::imshow("Full output", image_curr_show);
  cv::waitKey(0);
#endif

  // Set up example generator.
  example_generator_->Reset(bbox_gt,
                           bbox_gt,
                           image_curr,
                           image_curr); // use the same image as initial step fine-tuning

  // data structures to invoke fine tune
  std::vector<cv::Mat> images;
  std::vector<cv::Mat> targets;
  std::vector<BoundingBox> bboxes_gt_scaled;
  std::vector<std::vector<cv::Mat> > candidates; 
  std::vector<std::vector<double> >  labels;

  // Generate true example.
  cv::Mat image;
  cv::Mat target;
  BoundingBox bbox_gt_scaled;
  example_generator_->MakeTrueExample(&image, &target, &bbox_gt_scaled);
  
  images.push_back(image);
  targets.push_back(target);
  bboxes_gt_scaled.push_back(bbox_gt_scaled);

  // Generate additional training examples through synthetic transformations.
  example_generator_->MakeTrainingExamples(FINE_TUNE_AUGMENT_NUM, &images,
                                           &targets, &bboxes_gt_scaled);
                                           
  std::vector<cv::Mat> this_frame_candidates;
  std::vector<double> this_frame_labels;

  // generate candidates and push to this_frame_candidates and this_frame_labels
  example_generator_->MakeCandidatesAndLabels(&this_frame_candidates, &this_frame_labels);

  // TODO: avoid the copying and just pass a vector of one frame's +/- candidates to train
  for(int i = 0; i< images.size(); i ++ ) {
    candidates.push_back(std::vector<cv::Mat>(this_frame_candidates)); // copy
    labels.push_back(std::vector<double>(this_frame_labels)); // copy
  }

  //Fine Tune!
  regressor_train_->TrainBatch(images,
                               targets,
                               bboxes_gt_scaled,
                               candidates,
                               labels,
                               -1); // -1 indicating fine tuning

}


void TrackerFineTune::ProcessTrackOutput(
    const size_t frame_num, const cv::Mat& image_curr, const bool has_annotation,
    const BoundingBox& bbox_gt, const BoundingBox& bbox_estimate_uncentered,
    const int pause_val) {
  cv::Mat full_output;
  image_curr.copyTo(full_output);

  if (has_annotation) {
    // Draw ground-truth bounding box of the target location (white).
    bbox_gt.DrawBoundingBox(&full_output);
  }

  // Draw estimated bounding box of the target location (red).
  bbox_estimate_uncentered.Draw(255, 0, 0, &full_output);

  // Show the image with the estimated and ground-truth bounding boxes.
  cv::namedWindow("Full output", cv::WINDOW_AUTOSIZE ); // Create a window for display.
  cv::imshow("Full output", full_output );                   // Show our image inside it.

  // Pause for pause_val milliseconds, or until user input (if pause_val == 0).
  cv::waitKey(pause_val);
}




TrackerTesterAlov::TrackerTesterAlov(const std::vector<Video>& videos,
                                     const bool save_videos,
                                     RegressorBase* regressor, Tracker* tracker,
                                     const std::string& output_folder) :
  TrackerManager(videos, regressor, tracker),
  output_folder_(output_folder),
  hrt_("Tracker"),
  total_ms_(0),
  num_frames_(0),
  save_videos_(save_videos),
  fps_(20)
{
}

void TrackerTesterAlov::VideoInit(const Video& video, const size_t video_num) {
  // Get the name of the video from the video file path.
  int delim_pos = video.path.find_last_of("/");
  const string& video_name = video.path.substr(delim_pos+1, video.path.length());
  printf("Video %zu: %s\n", video_num + 1, video_name.c_str());

  // Open a file for saving the tracking output.
  const string& output_file = output_folder_ + "/" + video_name;
  output_file_ptr_ = fopen(output_file.c_str(), "w");

  if (save_videos_) {
    // Make a folder to save the tracking videos.
    const string& video_out_folder = output_folder_ + "/videos";
    boost::filesystem::create_directories(video_out_folder);

    // Get the size of the images that will be saved.
    cv::Mat image;
    BoundingBox box;
    video.LoadFrame(0, false, false, &image, &box);

    // Open a video_writer object to save the tracking videos.
    const string video_out_name = video_out_folder + "/Video" + num2str(static_cast<int>(video_num)) + ".avi";
    video_writer_.open(video_out_name, CV_FOURCC('M','J','P','G'), fps_, image.size());
  }
}

void TrackerTesterAlov::SetupEstimate() {
  // Record the time before starting to track.
  hrt_.reset();
  hrt_.start();
}

void TrackerTesterAlov::ProcessTrackOutput(
    const size_t frame_num, const cv::Mat& image_curr, const bool has_annotation,
    const BoundingBox& bbox_gt, const BoundingBox& bbox_estimate,
    const int pause_val) {
  // Stop the timer and print the time needed for tracking.
  hrt_.stop();
  const double ms = hrt_.getMilliseconds();

  // Update the total time needed for tracking.  (Other time is used to save the tracking
  // output to a video and to write tracking data to a file for evaluation purposes).
  total_ms_ += ms;
  num_frames_++;

  // Get the tracking output.
  const double width = fabs(bbox_estimate.get_width());
  const double height = fabs(bbox_estimate.get_height());
  const double x_min = std::min(bbox_estimate.x1_, bbox_estimate.x2_);
  const double y_min = std::min(bbox_estimate.y1_, bbox_estimate.y2_);

  // Save the trackign output to a file inthe appropriate format for the ALOV dataset.
  fprintf(output_file_ptr_, "%zu %lf %lf %lf %lf\n", frame_num + 1, x_min, y_min, width,
          height);

  if (save_videos_) {
    cv::Mat full_output;
    image_curr.copyTo(full_output);

    if (has_annotation) {
      // Draw ground-truth bounding box (white).
      bbox_gt.DrawBoundingBox(&full_output);
    }

    // Draw estimated bounding box on image (red).
    bbox_estimate.Draw(255, 0, 0, &full_output);

    // Save the image to a tracking video.
    video_writer_.write(full_output);
  }
}

void TrackerTesterAlov::PostProcessVideo() {
  // Close the file that saves the tracking data.
  fclose(output_file_ptr_);
}

void TrackerTesterAlov::PostProcessAll() {
  printf("Finished tracking %zu videos with %d total frames\n", videos_.size(), num_frames_);

  // Compute the mean tracking time per frame.
  const double mean_time_ms = total_ms_ / num_frames_;
  printf("Mean time: %lf ms\n", mean_time_ms);
}
