cmake_minimum_required(VERSION 2.8)
project(MDNet_C++_ROIPOOL)

SET(GCC_PEFORMANCE_COMPILE_FLAGS " -Ofast -msse2 -march=native -fopenmp -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${GCC_PEFORMANCE_COMPILE_FLAGS}")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "No build type selected, default to ${CMAKE_BUILD_TYPE}")
endif()
message ("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

find_package(Boost COMPONENTS system filesystem regex REQUIRED)
find_package(Protobuf)
if(${PROTOBUF_FOUND})
    message("!!found Google Proto Buffer library.")
else()
    message("!!Google Proto Buffer library not found.")
endif()

FIND_PACKAGE(GSL REQUIRED)

if (GSL_FOUND)
	message("--- found GSL, include directories.")
	INCLUDE_DIRECTORIES(${GSL_INCLUDE_DIRS} ${GSLCBLAS_INCLUDE_DIRS})
	SET(GSLLIB ${LIBS} ${GSL_LIBRARIES} ${GSLCBLAS_LIBRARIES})
else ()
	message("--- could not find GSL")
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
find_package(TinyXML REQUIRED)

find_package( OpenCV REQUIRED )
message("Open CV version is ${OpenCV_VERSION}")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
# Note: If can't find CUDA, please uncomment the below line and set the path manually
# set(CUDA_INCLUDE_DIRS /path_to_cuda/include)

find_package(Caffe REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY
message("Caffe_DIR is ${Caffe_DIR}")	#specify Caffe_DIR in /cmake/Modules/FindCaffe.cmake
# Note: If can't find Caffe, please uncomment the below line and set the path manually
# set(Caffe_DIR /home/jimxing/Downloads/install_caffe/caffe)
# set(Caffe_INCLUDE_DIRS /home/jimxing/Downloads/install_caffe/caffe/include)

set(GLOG_LIB glog)

find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
    message("-- Found Eigne3: ${EIGEN3_INCLUDE_DIR}")
    include_directories(${EIGEN3_INCLUDE_DIR})
else ()
    message("--- could not find Eigen3")
endif ()

FIND_PACKAGE(trax REQUIRED COMPONENTS core opencv)
if (TRAX_FOUND)
    message ("-- Found trax:")
    message ("-- TRAX_LIBRARIES: ${TRAX_LIBRARIES}")
    message ("-- TRAX_INCLUDE_DIRS: ${TRAX_INCLUDE_DIRS}")
    message ("-- TRAX_LIBRARY_DIRS: ${TRAX_LIBRARY_DIRS}")
else ()
    message ("-- could not find trax")
endif()
INCLUDE_DIRECTORIES(AFTER ${TRAX_INCLUDE_DIRS})
LINK_DIRECTORIES(AFTER ${TRAX_LIBRARY_DIRS})

file (GLOB_RECURSE SOURCE_FILES
"src/helper/*.cpp"
"src/train/example_generator.cpp"
"src/train/tracker_trainer.cpp"
"src/train/tracker_trainer_multi_domain.cpp"
"src/loader/*.cpp"
"src/network/*.cpp"
"src/tracker/*.cpp"

"src/helper/*.h"
"src/train/example_generator.h"
"src/train/tracker_trainer.h"
"src/train/tracker_trainer_multi_domain.h"
"src/loader/*.h"
"src/network/*.h"
"src/tracker/*.h"
"src/native/vot.h"

"src/rapidxml/*.hpp"
)

add_library (${PROJECT_NAME} ${SOURCE_FILES})

# Add src to include directories.
include_directories(src)

# Add all files.
#file(GLOB_RECURSE hdrs src/*.h*)
#file(GLOB_RECURSE srcs src/*.cpp)
#add_library (${PROJECT_NAME} ${srcs} ${hdrs})

# message ("** OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
# message ("** Caffe_INCLUDE_DIRS: ${Caffe_INCLUDE_DIRS}")
# message ("** GLOG_INCLUDE_DIRS: ${GLOG_INCLUDE_DIRS}")
# message ("** PROTOBUF_INCLUDE_DIRS: ${PROTOBUF_INCLUDE_DIRS}")
# message ("** GSL_INCLUDE_DIRS: ${GSL_INCLUDE_DIRS}")
# message ("** Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")

# add_executable (test_tracker_vot src/test/test_tracker_vot.cpp)
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${GLOG_LIB} ${PROTOBUF_LIBRARIES} ${GSLLIB})
# # Note: If can't find trax, please download trax and build it, then uncomment the below line and set the path manually
# target_link_libraries(${PROJECT_NAME} /home/jimxing/proj/trax/build/libtrax.so)
# target_link_libraries (test_tracker_vot ${PROJECT_NAME})

# add_executable (test_tracker_no_middle_batch_single_no_pool_avg src/test/test_tracker_vot_gmd_finetune.cpp)
# target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${Boost_LIBRARIES} ${GLOG_LIB} ${PROTOBUF_LIBRARIES} ${GSLLIB} ${EIGEN3_LIBS})
# # Note: If can't find trax, please download trax and build it, then uncomment the below line and set the path manually
# target_link_libraries(${PROJECT_NAME} /home/jimxing/proj/trax/build/libtrax.so)
# target_link_libraries (test_tracker_no_middle_batch_single_no_pool_avg ${PROJECT_NAME})

add_executable (trax_tracker_mdnet_roipool src/test/test_tracker_vot_gmd_finetune_trax.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${Boost_LIBRARIES} ${GLOG_LIB} ${PROTOBUF_LIBRARIES} ${GSLLIB} ${EIGEN3_LIBS} ${TRAX_LIBRARIES})
target_link_libraries (trax_tracker_mdnet_roipool ${PROJECT_NAME})

# add_executable (test_tracker_alov src/test/test_tracker_alov.cpp)
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${GLOG_LIB} ${PROTOBUF_LIBRARIES} ${GSLLIB})
# target_link_libraries (test_tracker_alov ${PROJECT_NAME})

# add_executable (save_videos_vot src/test/save_videos_vot.cpp)
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${GLOG_LIB} ${PROTOBUF_LIBRARIES} ${GSLLIB})
# target_link_libraries (save_videos_vot ${PROJECT_NAME})

# add_executable (train src/train/train.cpp)
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${TinyXML_LIBRARIES} ${GLOG_LIB} ${PROTOBUF_LIBRARIES} ${GSLLIB})
# target_link_libraries (train ${PROJECT_NAME})

add_executable (train_mdnet_roipool src/train/train_multi_domain.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${TinyXML_LIBRARIES} ${GLOG_LIB} ${PROTOBUF_LIBRARIES} ${GSLLIB})
target_link_libraries (train_mdnet_roipool ${PROJECT_NAME})

# add_executable (show_tracker_vot src/visualizer/show_tracker_vot.cpp)
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${Boost_LIBRARIES} ${GLOG_LIB} ${PROTOBUF_LIBRARIES} ${GSLLIB})
# target_link_libraries (show_tracker_vot ${PROJECT_NAME})

# add_executable (show_tracker_vot_gmd src/visualizer/show_tracker_vot_gmd.cpp)
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${Boost_LIBRARIES} ${GLOG_LIB} ${PROTOBUF_LIBRARIES} ${GSLLIB})
# target_link_libraries (show_tracker_vot_gmd ${PROJECT_NAME})

add_executable (show_mdnet_roipool src/visualizer/show_tracker_vot_gmd_finetune.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${Boost_LIBRARIES} ${GLOG_LIB} ${PROTOBUF_LIBRARIES} ${GSLLIB} ${EIGEN3_LIBS})
target_link_libraries (show_mdnet_roipool ${PROJECT_NAME})

add_executable (UnitTest src/UnitTest/unit_test.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${TinyXML_LIBRARIES} ${GLOG_LIB} ${PROTOBUF_LIBRARIES} ${GSLLIB} ${EIGEN3_LIBS})
target_link_libraries (UnitTest ${PROJECT_NAME})


# add_executable (show_tracker_otb src/visualizer/show_tracker_otb.cpp)
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${Boost_LIBRARIES} ${GLOG_LIB} ${PROTOBUF_LIBRARIES} ${GSLLIB})
# target_link_libraries (show_tracker_otb ${PROJECT_NAME})

# add_executable (show_tracker_alov src/visualizer/show_tracker_alov.cpp)
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${Boost_LIBRARIES} ${GLOG_LIB} ${PROTOBUF_LIBRARIES} ${GSLLIB})
# target_link_libraries (show_tracker_alov ${PROJECT_NAME})

# add_executable (show_imagenet src/visualizer/show_imagenet.cpp)
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TinyXML_LIBRARIES} ${PROTOBUF_LIBRARIES} ${GSLLIB})
# target_link_libraries (show_imagenet ${PROJECT_NAME})

# add_executable (show_alov src/visualizer/show_alov.cpp)
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PROTOBUF_LIBRARIES} ${GSLLIB})
# target_link_libraries (show_alov ${PROJECT_NAME})


