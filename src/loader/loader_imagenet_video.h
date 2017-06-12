#ifndef LOADER_IMAGENET_VIDEO_H
#define LOADER_IMAGENET_VIDEO_H

#include "helper/Common.h"
#include "helper/CommonCV.h"
#include "helper/bounding_box.h"
#include "loader/video_imagenet.h"
#include "../rapidxml/rapidxml.hpp"
#include "../rapidxml/rapidxml_utils.hpp"
#include <assert.h>
#include "helper/helper.h"

using namespace rapidxml;

class LoaderImageNetVideo {
public:
  // data_folder: containing ILSVRC2015_VID_train_0000/ ... image files
  // annotation_folder: containing ILSVRC2015_VID_train_0000/ ... annotation xmls to parse
  LoaderImageNetVideo(const string & data_folder, const string & annotation_folder);

  std::vector<VideoImageNet> get_videos() const { return videos_; }

  void LoadInfoOneVideo(string & video_data_folder, string & video_annotation_folder, VideoImageNet &video);

  void ParseAnnotationXML(string & annotation_file, vector<TrackObject> &track_objs);

protected:
  std::vector<VideoImageNet> videos_;
};

#endif // VIDEO_LOADER_H

