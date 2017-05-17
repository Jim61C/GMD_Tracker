#include "tracker_manager.h"

#include <string>

#include "helper/helper.h"
#include "train/tracker_trainer.h"
#include <algorithm>

using std::string;

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

    // Perform any pre-processing steps on this video
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

      // After estimation, update state, previous frame, new bbox priors etc.
      tracker_->UpdateState(image_curr, bbox_estimate_uncentered, regressor_, tracker_->GetInternelCurFrame() == total_num_frames_ - 1);

      
      // Process the output (e.g. visualize / save results).
      ProcessTrackOutput(frame_num, image_curr, has_annotation, bbox_gt,
                           bbox_estimate_uncentered, pause_val);

    }
    PostProcessVideo(video_num);
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
  total_num_frames_ = video.all_frames.size();
  printf("Video: %zu\n", video_num);
}


// Tracker With Fine Tuning Ability
TrackerFineTune::TrackerFineTune(const std::vector<Video>& videos,
                                RegressorBase* regressor, Tracker* tracker,
                                bool save_videos,
                                const std::string output_folder,
                                bool show_result) :
  TrackerManager(videos, regressor, tracker),
  save_videos_(save_videos),
  output_folder_(output_folder),
  hrt_("TrackerFineTune"),
  total_ms_(0),
  num_frames_(0),
  fps_(20),
  show_result_(show_result)

{
  if (output_folder_.back() != '/') {
    output_folder_ += '/';
  }
}


void TrackerFineTune::VideoInit(const Video& video, const size_t video_num) {
  // update total_num_frames_
  total_num_frames_ = video.all_frames.size();

  printf("In TrackerFineTune, Video Init: %zu\n", video_num);
    
  // set up video saver and output path
  string video_name = video.path;
  std::replace(video_name.begin(), video_name.end(), '/', '_');
  printf("Set up saver for Video %zu, save name: %s\n", video_num + 1, video_name.c_str());

  // Open a file for saving the tracking output.
  const string& output_file = output_folder_ + video_name;
  // output_file_ptr_ = fopen(output_file.c_str(), "w");

  if (save_videos_) {
    // Make a folder to save the tracking videos.
    const string& video_out_folder = output_folder_ + "videos";
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


void TrackerFineTune::ProcessTrackOutput(
    const size_t frame_num, const cv::Mat& image_curr, const bool has_annotation,
    const BoundingBox& bbox_gt, const BoundingBox& bbox_estimate_uncentered,
    const int pause_val) {
    
  cout << "process tracker output for frame " << frame_num << endl;
  hrt_.stop();
  const double ms = hrt_.getMilliseconds();

  // Update the total time needed for tracking before visualise and save
  total_ms_ += ms;

  // Increment number of frames tracked
  num_frames_ ++;

  printf("Track frame %zu, time spent: %lf ms\n\n", frame_num, ms);

  // Visualise
  cv::Mat full_output;
  image_curr.copyTo(full_output);

  if (has_annotation) {
    // Draw ground-truth bounding box of the target location (white).
    bbox_gt.DrawBoundingBox(&full_output);
  }

  // Draw estimated bounding box of the target location (red).
  bbox_estimate_uncentered.Draw(255, 0, 0, &full_output);

  if (show_result_) {
    // Show the image with the estimated and ground-truth bounding boxes.
    cv::namedWindow("Full output", cv::WINDOW_AUTOSIZE ); // Create a window for display.
    cv::imshow("Full output", full_output );                   // Show our image inside it.

    // Pause for pause_val milliseconds, or until user input (if pause_val == 0).
    cv::waitKey(pause_val);
  }

  // write to video
  if (save_videos_) {
    // Save the image to a tracking video.
    video_writer_.write(full_output);
  }
}


void TrackerFineTune::PostProcessVideo(size_t video_num) {
  // Close the file that saves the tracking data.
  // fclose(output_file_ptr_)

  // clear all the storage in the tracker
  tracker_->Reset(regressor_);

  // report the time and average fps
  const double mean_fps = num_frames_ / (total_ms_ / 1000.0);
  printf("Video %zu Mean fps: %lf ms\n", video_num, mean_fps);

  // reset counters
  total_ms_ = 0;
  num_frames_ = 0;
}

void TrackerFineTune::SetupEstimate() {
  // Record the time before starting to track.
  hrt_.reset();
  hrt_.start();
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
  total_num_frames_ = video.all_frames.size();
  
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

void TrackerTesterAlov::PostProcessVideo(size_t video_num) {
  // Close the file that saves the tracking data.
  fclose(output_file_ptr_);
}

void TrackerTesterAlov::PostProcessAll() {
  printf("Finished tracking %zu videos with %d total frames\n", videos_.size(), num_frames_);

  // Compute the mean tracking time per frame.
  const double mean_time_ms = total_ms_ / num_frames_;
  printf("Mean time: %lf ms\n", mean_time_ms);
}
