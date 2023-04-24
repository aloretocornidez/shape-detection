// Opencv header includes
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/opencv.hpp>

// Cuda header includes
#include <cuda_runtime.h>
#include <cuda.h>

// Additional Includes
#include "hough-transform.hpp"
#include "interface.hpp"
#include <iostream>

// Namespace function declarations
using namespace cv;
using std::cerr;
using std::cout;
using std::endl;

int main(int argc, char **argv)
{

  // Usage for the program
  if (argc < 2)
  {

    cerr << "Usage: {programName} path/to/image" << endl; // (0: CPU | 1: GPU)" << endl;
    exit(-1);
  }
  int method = atoi(argv[2]);
  if ((method != 0) && (method != 1))
  {
    cerr << "Method must be (0: CPU | 1: GPU)" << endl;
    exit(-1);
  }

  // Read the image file
  Mat inputImage = imread(argv[1], IMREAD_GRAYSCALE);

  // Check for failure when opening the image.
  if (inputImage.empty())
  {

    cout << "Could not open or find the image with path:" << argv[1] << endl;
    return -1;
  }
  else
  {
    cout << "Image loaded: " << argv[1] << endl;
  }

  cv::threshold(inputImage, inputImage, 100, 255, THRESH_BINARY);

  // Vector containing the coordinate values of the circles found.
  std::vector<Vec3f> circles;

  houghTransform(inputImage, circles, method);

  /*
   *
   * Drawing circles on the image
   *
   */
  // Were circles found?
  if (!circles.empty())
  {
    std::cout << "Circles Found: " << circles.size() << std::endl;

    for (int i = 0; i < (int)circles.size(); i++)
    {

      Vec3i cir = circles[i];

      circle(inputImage, Point(cir[0], cir[1]), cir[2], Scalar(0, 0, 0), 2, LINE_AA);
      circle(inputImage, Point(cir[0], cir[1]), 1, Scalar(128, 128, 128), 2, LINE_AA);
    }
    // imshow("Test Circles found", inputImage);
    // waitKey();
  }
  else
  {
    std::cout << "No circles found" << std::endl;
  }

  return 0;
}