#include "homework-6.hpp"
// #include "erosion.hpp"
// #include "dilation.hpp"

#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{

  // Usage for the program
  if (argc != 2)
  {
    // std::cout << "Usage: {programName} path/to/image" << std::endl;
    return -1;
  }

  std::cout << "Running Main driver file." << std::endl;

  // Read the image file
  cv::Mat inputImage = imread(argv[1], cv::IMREAD_GRAYSCALE);

  // Check for failure when opening the image.
  if (inputImage.empty())
  {
    std::cout << "Could not open or find the image" << std::endl;
    return -1;
  }





  return 0;
}
