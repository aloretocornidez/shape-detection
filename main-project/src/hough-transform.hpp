#ifndef __HOUGH_TRANSFORM__
#define __HOUGH_TRANSFORM__

#include <opencv2/opencv.hpp>

/*
 * Hough Transform Functions
 */
// Perform the Hough Transform, the final argument determines which kernel is used.
/* The CPU Kernel currently does not generate an R-Table but the GPU kernels to. The R-Table is interpreted on the cpu.*/
void houghTransform(cv::Mat &grayscaleInputImage, std::vector<cv::Vec3f> &circles, int method);

// Perform the Hough Transform on the CPU.
void cpuKernelHoughTransform(cv::Mat &srcImage, std::vector<cv::Vec3f> &srcCircles);

// Appends circles to the vector of circles contained in the image. Uses the R-Table generated by the gpuKernels.
void parseRTable(std::vector<cv::Vec3f> &circles, unsigned int *rTable, int minimumRadius, int maximumRadius, int imageRows, int imageColumns);

/*
 * Array Add Functions
 */
// Perform an element-wise addition of two arrays. Output stored in array_1
void cudaAddKernel(int size, int *array_1, int *array_2);

#endif
