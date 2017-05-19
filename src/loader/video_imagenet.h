#ifndef VIDEO_IMAGENET_H
#define VIDEO_IMAGENET_H

#include "track_object.h"

class VideoImageNet {

public:
  
  // check if the given two frames have track objects in common
  void CheckTwoFramesCommonTrackObject(const int frame1, const int frame2, vector<int> &common_trackids) const;

  // at least one pair of t-1 to t same track object
  bool hasEnoughAnnotation() const;

  // get the track object ids appear in given frame_num
  void GetFrameTrackids(const int frame_num, vector<int> &trackids) const;

  // For a given frame num, find an annotation if it exists, and return true.
  // Otherwise return false.
  // If load_only_annotation = false, also load the image.
  // If draw_bounding_box = true, draw the annotation on the image.
  bool LoadFrame(const int frame_num,
                const int track_obj_id,
                const bool draw_bounding_box,
                const bool load_only_annotation,
                cv::Mat* image,
                BoundingBox* box) const;

  // Path to the folder containing the image files for this video.
  std::string path;

  // Name of all image files for this video (must be appended to path).
  std::vector<std::string> all_frames;

  // track objects in each frame
  std::vector<std::vector<TrackObject> > frame_objects;

private:
    // given frame number and track object id, get the annotation if it exist in that frame, otherwise return false
    bool FindTrackObjectAnnotation(const int frame_num, const int track_obj_id, BoundingBox * box) const;
};

#endif