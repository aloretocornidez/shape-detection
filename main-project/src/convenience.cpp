#include <opencv2/opencv.hpp>
#include "convenience.hpp"



void openWindow(cv::String windowName, cv::Mat *image)
{

  cv::namedWindow(windowName); // Create a window

  cv::imshow(windowName, *image); // Show our image inside the created window.

  cv::waitKey(0); // Wait for any keystroke in the window

  cv::destroyWindow(windowName); // destroy the created window
}