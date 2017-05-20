#include "loader_imagenet_video.h"
#include <stdlib.h> 

LoaderImageNetVideo::LoaderImageNetVideo(const string & data_folder, const string & annotation_folder) {
    vector<string> sub_folders;
    find_subfolders(data_folder, &sub_folders);

    for (int i =0; i < sub_folders.size(); i++) {
        string this_sub_folder = sub_folders[i];
        vector<string> video_folders;
        find_subfolders(data_folder + "/" + this_sub_folder, &video_folders);

        // parse each video_folder
        for (int j = 0; j < video_folders.size(); j++) {
            VideoImageNet video;
            string this_video_data_folder_path = data_folder + "/" + this_sub_folder + "/" + video_folders[j];
            string this_video_annotation_folder_path = annotation_folder + "/" + this_sub_folder + "/" + video_folders[j];
            LoadInfoOneVideo(this_video_data_folder_path, this_video_annotation_folder_path, video);
            videos_.push_back(video);
            // cout << "finish load video: " << video.path << ", n_frames: " << video.all_frames.size() << endl;
        }

        cout << "finish load video from set: " << data_folder + "/" + this_sub_folder << endl;
    }
}


void LoaderImageNetVideo::LoadInfoOneVideo(string & video_data_path, string & video_annotation_path, VideoImageNet &video) {
    const boost::regex image_filter(".*\\.JPEG");
    find_matching_files(video_data_path, image_filter, &video.all_frames);
    // make sure no / in video_data_path
    if (video_data_path.back() == '/') {
        video_data_path = video_data_path.substr(0, video_data_path.length()-1);
    }
    video.path = video_data_path;

    if (video_annotation_path.back() != '/') {
        video_annotation_path += '/';
    }
    for (int i = 0; i < video.all_frames.size(); i ++) {
        string frame_name = video.all_frames[i];
        string frame_idx_str = frame_name.substr(0, frame_name.find(".JPEG"));
        string frame_corres_annotation_path = video_annotation_path + frame_idx_str + ".xml";
        vector<TrackObject> this_frame_track_objs;
        ParseAnnotationXML(frame_corres_annotation_path, this_frame_track_objs);
        video.frame_objects.push_back(this_frame_track_objs);
    }

    assert (video.all_frames.size() == video.frame_objects.size());
}

void LoaderImageNetVideo::ParseAnnotationXML(string & annotation_file, vector<TrackObject> &track_objs) {
    rapidxml::file<> xmlFile(annotation_file.c_str()); // Default template is char
    rapidxml::xml_document<> doc;
    doc.parse<0>(xmlFile.data());

    assert(strcmp(doc.first_node()->name(), "annotation") == 0);

    xml_node<> *annotation_node = doc.first_node();
    int W, H;

    for (xml_node<> *child = annotation_node->first_node(); child; child = child->next_sibling()) {
      // do stuff with child
      if (strcmp(child->name(), "object") == 0) {
        TrackObject this_obj;
        for (xml_node<> *track_obj_node = child->first_node(); track_obj_node; track_obj_node = track_obj_node->next_sibling()) {
          if (strcmp(track_obj_node->name(), "trackid") == 0) {
            this_obj.trackid_ = atoi(track_obj_node->value());
          }
          else if (strcmp(track_obj_node->name(), "name") == 0) {
            this_obj.name_ = std::string(track_obj_node->value());
          }
          else if (strcmp(track_obj_node->name(), "occluded") == 0) {
            // cout << "\t" << "occluded: " << track_obj_node->value() << endl;
          }
          else if (strcmp(track_obj_node->name(), "generated") == 0) {
            // cout << "\t" << "generated: " << track_obj_node->value() << endl;
          }
          else if (strcmp(track_obj_node->name(), "bndbox") == 0) {
            xml_node<> *xmax_node = track_obj_node->first_node();
            xml_node<> *xmin_node = xmax_node->next_sibling();
            xml_node<> *ymax_node = xmin_node->next_sibling();
            xml_node<> *ymin_node = ymax_node->next_sibling();

            int x1 = atoi(xmin_node->value());
            int y1 = atoi(ymin_node->value());
            int x2 = atoi(xmax_node->value());
            int y2 = atoi(ymax_node->value());

            this_obj.annotation_ = BoundingBox(x1, y1, x2, y2);
          }
        }

        track_objs.push_back(this_obj);
      }
      else if (strcmp(child->name(), "size") == 0) {
        // record the W and H
        xml_node<> *width_node = child->first_node();
        assert(strcmp(width_node->name(), "width") == 0);

        xml_node<> *height_node = width_node->next_sibling();
        assert(strcmp(height_node->name(), "height") == 0);

        W = atoi(width_node->value());
        H = atoi(width_node->value());
      }
    }

    // set W and H for all objects
    for (int i = 0; i <track_objs.size();i++) {
        track_objs[i].frame_h_ = H;
        track_objs[i].frame_w_ = W;
    }
}