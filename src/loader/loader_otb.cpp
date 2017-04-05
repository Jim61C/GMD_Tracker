#include "loader_otb.h"
#include "helper/Common.h"
#include "helper/CommonCV.h"
#include "helper/helper.h"
#include <algorithm> // std::min

namespace bfs = boost::filesystem;

#define DEBUGGING

char parseDelimiter(string str) {
    char delimiters [3] = {';', '\t', ','};

    for (int i = 0;i< 3; i++) {
        char this_char = delimiters[i];
        if (str.find(this_char) != std::string::npos) {
            return this_char;
        }
    }

    return ' ';
}

LoaderOTB::LoaderOTB(const std::string& otb_folder) {
    if (!bfs::is_directory(otb_folder)) {
        printf("Error - %s is not a valid directory!\n", otb_folder.c_str());
        return;
    }

    // Find all video subcategories.
    vector<string> videos;
    find_subfolders(otb_folder, &videos);
    string otb_folder_complete;

    if (otb_folder.back() != '/') {
        otb_folder_complete = otb_folder + '/';
    }
    else {
        otb_folder_complete = otb_folder;
    }

    printf("Found %zu videos...\n", videos.size());
    for (size_t i = 0; i < videos.size(); ++i) {
        const string& video_name = videos[i];
        string video_path = otb_folder_complete + video_name;

        printf("Loading video: %s\n", video_name.c_str());

        Video video;
        if (video_path.back() != '/') {
            video_path += '/';
        }
        video.path = video_path + "img";

        // Find all image files
        const boost::regex image_filter(".*\\.jpg");
        find_matching_files(video.path, image_filter, &video.all_frames);

        // Open the annotation file.
        const string& bbox_groundtruth_path = video_path + "/groundtruth_rect.txt";
        ifstream instream(bbox_groundtruth_path.c_str());

        char *delimiter = NULL;
        char parsed_delimiter;
        string this_line;
        int frame_num = 0;
        double x, y, w, h;

        while (!instream.eof()) {

            // Read the annotation data.
            getline(instream, this_line);

            // only process it if this_line is not zero length
            if (this_line.length() != 0) {
                // parse the delimiter if NULL
                if (delimiter == NULL) {
                    parsed_delimiter = parseDelimiter(this_line);
                    delimiter = &parsed_delimiter;
                }

                // if not space delimited, tab, comma or semicolumn delimited, convert to space delimited
                if ((*delimiter) != ' ') {
                    istringstream from_str_delimiter(this_line);
                    ostringstream out_str;
                    string token;
                    while (getline(from_str_delimiter, token, *delimiter)) {
                        out_str << token << ' ';
                    }

                    this_line = out_str.str();
                }

                
                istringstream iss_this_line_space_delimited(this_line);
                iss_this_line_space_delimited >> x;
                iss_this_line_space_delimited >> y;
                iss_this_line_space_delimited >> w;
                iss_this_line_space_delimited >> h;

                // Convert to bounding box format.
                Frame frame;
                frame.frame_num = frame_num;
                BoundingBox& bbox = frame.bbox;
                bbox.x1_ = x;
                bbox.y1_ = y;
                bbox.x2_ = x + w;
                bbox.y2_ = y + h;

                cout << frame_num << ": " << bbox.x1_ << ", " << bbox.y1_ << ", " << bbox.x2_ << ", " << bbox.y2_ << endl;

                // Increment the frame number.
                frame_num++;

                video.annotations.push_back(frame);   
            }
        } // Processed annotation file

        // to be safe, use min of (video.annotations.size(), video.all_frames.size())
        int n_frames = std::min(video.annotations.size(), video.all_frames.size());
        video.annotations.erase(video.annotations.begin() + n_frames, video.annotations.end());
        video.all_frames.erase(video.all_frames.begin() + n_frames, video.all_frames.end());

        videos_.push_back(video);
    } // Processed all videos
}