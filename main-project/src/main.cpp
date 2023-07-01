// Opencv header includes
#include <opencv2/opencv.hpp>

// Cuda header includes
#include <cuda_runtime.h>
// #include <cuda.h>

// Additional Includes
#include "hough-transform.hpp"
#include <iostream>

// Namespace function declarations
using namespace cv;
using std::cerr;
using std::cout;
using std::endl;

// Methods available to execute the hough transform.
#define HOUGH_METHODS_AVAILABLE 3

// Default specified method for the hough transform. CPU=0. GPU=1
#define DEFAULT_METHOD 1

// Parses the parameters that are passed arguments into the function.
static void argumentParsing(int &argc, char *const argv[], int &method)
{
  // Usage for the program
  if (argc < 2)
  {

    cerr << "Usage: {programName} path/to/image" << endl; // (0: CPU | 1: GPU)" << endl;
    exit(-1);
  }

  if (argc == 3)
  {
    method = atoi(argv[2]);
    std::cout << "Parsed method is: " << method << std::endl;
  }
  else
  {
    method = DEFAULT_METHOD;
    std::cout << "No method specified, using method (" << method << ") to execute the hough-transform." <<  std::endl;

  }

  if (method > HOUGH_METHODS_AVAILABLE || method < 0)
  {
    cerr << "Method must be less than: " << HOUGH_METHODS_AVAILABLE << endl;
    exit(-1);
  }
}

// Draws cricles on an image from a given rTable.
void drawCircles(cv::Mat &inputImage, std::vector<cv::Vec3f> &circles)
{
  // Were circles found?
  if (!circles.empty())
  {
    std::cout << "Circles Found: " << circles.size() << std::endl;

    for (int i = 0; i < (int)circles.size(); i++)
    {

      Vec3i cir = circles[i];

      circle(inputImage, Point(cir[0], cir[1]), cir[2], Scalar(128, 128, 128), 2, LINE_AA);
      circle(inputImage, Point(cir[0], cir[1]), 1, Scalar(128, 128, 128), 2, LINE_AA);
    }
    imwrite("output.jpg", inputImage);
    // waitKey();
  }
  else
  {
    std::cout << "No circles found" << std::endl;
  }
}

int main(int argc, char *argv[])
{
  std::cout << "Main started" << std::endl;
  int executionMethod = -1;
  argumentParsing(argc, argv, executionMethod);

  std::cout << "Method after function is: " << executionMethod << std::endl;

  // Read the image file
  std::cout << "Reading Input image: " << argv[1] << std::endl;
  Mat inputImage = imread(argv[1]);

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

  // cv::threshold(inputImage, inputImage, 100, 255, THRESH_BINARY);

  // Vector containing the coordinate values of the circles found.
  std::vector<Vec3f> circles;

  // Execute the hough transform.
  std::cout << "Executing 'hough-transform' with method: " << executionMethod << std::endl;
  houghTransform(inputImage, circles, executionMethod);

  // Drawing circles on the image
  drawCircles(inputImage, circles);

  return 0;
}
