cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME "shape-detection-wrapper")

project(${PROJECT_NAME})


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



add_subdirectory(image-processing)
add_subdirectory(main)


# set(OPENCV_CUDA_SAMPLES_REQUIRED_DEPS
#   opencv_core
#   opencv_flann
#   opencv_imgproc
#   opencv_imgcodecs
#   opencv_videoio
#   opencv_highgui
#   opencv_ml
#   opencv_video
#   opencv_objdetect
#   opencv_features2d
#   opencv_calib3d
#   opencv_superres
#   opencv_cudaarithm
#   opencv_cudafilters
#   opencv_cudawarping
#   opencv_cudaimgproc
#   opencv_cudafeatures2d
#   opencv_cudaoptflow
#   opencv_cudabgsegm
#   opencv_cudastereo
#   opencv_cudaobjdetect)
# ocv_check_dependencies(${OPENCV_CUDA_SAMPLES_REQUIRED_DEPS})

# add_executable(${PROJECT_NAME})