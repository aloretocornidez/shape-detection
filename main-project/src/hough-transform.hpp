#ifndef __HOUGH_TRANSFORM__
#define __HOUGH_TRANSFORM__

#include <opencv2/core.hpp>

/*
 * Hough Transform Functions
 */
// Perform the Hough Transform, the final argument determines which kernel is used.
void houghTransform(cv::Mat &grayscaleInputImage, std::vector<cv::Vec3f> &circles, int method);
// Perform the Hough Transform on the CPU.
void cpuKernelHoughTransform(cv::Mat &srcImage, std::vector<cv::Vec3f> &srcCircles);

/*
 * Array Add Functions
 */
// Perform an element-wise addition of two arrays. Output stored in array_1
void cudaAddKernel(int size, int *array_1, int *array_2);

#endif