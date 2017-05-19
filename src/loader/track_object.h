#ifndef TRACK_OBJECT_H
#define TRACK_OBJECT_H

#include "../helper/Common.h"
#include "../helper/bounding_box.h"

struct TrackObject{
    int trackid_;
    string name_;
    int frame_w_;
    int frame_h_;
    BoundingBox annotation_;
};

#endif