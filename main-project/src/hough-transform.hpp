#ifndef __HOUGH_TRANSFORM__
#define __HOUGH_TRANSFORM__

#include <opencv2/core.hpp>

// Perform the Hough Transform
void cudaHoughTransform(cv::Mat& grayscaleInputImage, cv::InputArray circles);


// Perform an element-wise addition of two arrays. Output stored in array_1
void cudaAddKernel(int size, int* array_1, int* array_2);

#endif