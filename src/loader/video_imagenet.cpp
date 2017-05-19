#include "video_imagenet.h"
#include <algorithm>

// For a given frame num, find an annotation if it exists, and return true.
// Otherwise return false.
// If load_only_annotation = false, also load the image.
// If draw_bounding_box = true, draw the annotation on the image.
bool VideoImageNet::LoadFrame(const int frame_num,
            const int track_obj_id,
            const bool draw_bounding_box,
            const bool load_only_annotation,
            cv::Mat* image,
            BoundingBox* box) const {
  
  
  const string& video_path = path;
  const vector<string>& image_files = all_frames;

  if (image_files.empty()) {
    printf("Error - no image files for video at path: %s\n", path.c_str());
    return false;
  } else if (frame_num >= image_files.size()) {
    printf("Cannot find frame: %d; only %zu image files were found at %s\n", frame_num, image_files.size(), path.c_str());
    return false;
  }

  // Load the image for this frame.
  if (!load_only_annotation) {
    const string& image_file = video_path + "/" + image_files[frame_num];
    *image = cv::imread(image_file);
    if (!image->data) {
        printf("Could not find file: %s\n", image_file.c_str());
        return false;
    }
  }

  // check if the track object id exist on this frame, if exist, then assign to *box
  const bool has_track_obj_id = FindTrackObjectAnnotation(frame_num, track_obj_id, box);

  // Draw the annotation (if it exists) on the image.
  if (!load_only_annotation && has_track_obj_id && draw_bounding_box) {
    box->DrawBoundingBox(image);
  }

  // return whether *box was assigned
  return has_track_obj_id;
}


bool VideoImageNet::FindTrackObjectAnnotation(const int frame_num, const int track_obj_id, BoundingBox * box) const {
    const vector<TrackObject> & track_objs = frame_objects[frame_num];
    for (int i = 0; i < track_objs.size(); i++) {
        if (track_objs[i].trackid_ == track_obj_id) {
            *box = track_objs[i].annotation_;
            return true;
        }
    }

    return false;
}

void VideoImageNet::GetFrameTrackids(const int frame_num, vector<int> &trackids) const {
    const vector<TrackObject> & track_objs = frame_objects[frame_num];
    for (int i = 0; i < track_objs.size(); i ++) {
        trackids.push_back(track_objs[i].trackid_);
    }
}


void VideoImageNet::CheckTwoFramesCommonTrackObject(const int frame1, const int frame2, vector<int> &common_trackids) const {
    vector<int> frame1_trackids;
    GetFrameTrackids(frame1, frame1_trackids);

    vector<int> frame2_trackids;
    GetFrameTrackids(frame2, frame2_trackids);

    std::sort(frame1_trackids.begin(), frame1_trackids.end());
    std::sort(frame2_trackids.begin(), frame2_trackids.end());

    std::set_intersection(frame1_trackids.begin(), frame1_trackids.end(), frame2_trackids.begin(), frame2_trackids.end(), 
    std::back_inserter(common_trackids));
}


bool VideoImageNet::hasEnoughAnnotation() const {
    if (frame_objects.size() < 2) {
        return false;
    }

    for (int i = 1; i < frame_objects.size(); i ++) {
        vector<int> neighbour_frame_common_trackids;
        CheckTwoFramesCommonTrackObject(i - 1, i, neighbour_frame_common_trackids);
        if (neighbour_frame_common_trackids.size() > 0) {
            return true;
        }
    }

    return false;
}