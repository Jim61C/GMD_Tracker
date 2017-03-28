#ifndef LOADER_OTB_H
#define LOADER_OTB_H
#include "video_loader.h"

class LoaderOTB: public VideoLoader {
public:
    LoaderOTB(const std::string& otb_folder);
};

#endif