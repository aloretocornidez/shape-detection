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
using namespace std;

int main(int argc, char **argv)
{

  // Usage for the program
  if (argc < 2)
  {

    cerr << "Usage: {programName} path/to/image" << endl; // (0: CPU | 1: GPU)" << endl;
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

  // Vector containing the coordinate values of the cirles found.
  std::vector<Vec3f> circles;

  cv::imshow("Before Manipulation | Main", inputImage);
  cv::waitKey();

  cudaHoughTransform(inputImage, circles);

  cv::imshow("After Manipulation | Main", inputImage);
  cv::waitKey();


  
#if 0


  float gaussianStdDev = 1.0;
  int gaussianKernelSize = 3;

  
  // Checks if the program is being run on the CPU or GPU.
  bool gpuRunning = atoi(argv[2]);
  if (gpuRunning)
  {

    cout << "Running GPU" << endl;

    // Initialize a gpu image holder object.
    cv::cuda::GpuMat imgGpu, gpuBlurredImage, circlesGpu;

    imgGpu.upload(inputImage);

    // cuda::cvtColor(imgGpu, gray, COLOR_BGR2GRAY);

    // Image Filtering
    auto gaussianFilter = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, {3, 3}, 1);
    gaussianFilter->apply(imgGpu, imgGpu);

    // Circle Detector
    auto circleDetection = cuda::createHoughCirclesDetector(1, 100, 120, 50, 1, 500, 5);
    circleDetection->detect(imgGpu, circlesGpu);

    circles.resize(circlesGpu.size().width);
    if (!circles.empty())
    {
      circlesGpu.row(0).download(Mat(circles).reshape(3, 1));
    }

    /*
    // // Copy the input image from the Host to the GPU
    // cout << "Copying image to gpu." << endl;
    // imgGpu.upload(inputImage);
    // cout << "Uploaded image." << endl;

    // // Apply a gaussian filter on the image
    // cout << "Applying gaussian filter to image." << endl;
    // auto gaussianFilter = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, {3, 3}, 1);
    // gaussianFilter->apply(imgGpu, gpuBlurredImage);
    // cout << "Gaussian filter applied to image." << endl;

    // // Conduct circle detection on the image
    // cout << "Conducting circle detection." << endl;
    // auto circleDetection = cuda::createHoughCirclesDetector(1, 1, 120, 50, 1, 50, 5);
    // circleDetection->detect(gpuBlurredImage, circlesGpu);
    // cout << "Circle detection complete: " << endl;

    // // Setting the length of the circles vector on the host to the same size as the GPU.
    // circles.resize(circlesGpu.size().width);

    // if (!circles.empty())
    // {
    //   // Copies the data from the GPU to the host using reshape to place the data in the correct order.
    //   circlesGpu.row(0).download(Mat(circles).reshape(3, 1));
    // }

    // else
    // {
    //   cout << "Circles is empty" << endl;
    // }

    // cout << "Circles Size: " << circles.size() << endl;

    // cv::cuda::HoughCirclesDetector::detect(gpuInputImage, circles);
    */
  }
  /// Running on CPU
  else
  {
    cout << "Running CPU" << endl;

    Mat inputImageBuffer;
    // cvtColor(inputImage, gray, COLOR_BGR2GRAY);
    inputImageBuffer = inputImage.clone();

    // medianBlur(inputImageBuffer, inputImageBuffer, kernelSize);
    GaussianBlur(inputImageBuffer, inputImageBuffer, Size(gaussianKernelSize, gaussianKernelSize), gaussianStdDev);
    // imshow("blurred", inputImageBuffer);
    // waitKey();

    // Running a CPU hough transform.
    HoughCircles(inputImageBuffer, circles, HOUGH_GRADIENT, 1,
                 1,              // change this value to detect circles with different distances to each other
                 300, 37, 1, 200 // change the last two parameters
                                 // (min_radius & max_radius) to detect larger circles
    );
  }

  Mat outputImage = Mat(inputImage.rows, inputImage.cols, CV_8UC3, Scalar(0, 0, 0));
  for (size_t i = 0; i < circles.size(); i++)
  {
    RNG rng(12345); // random number

    int b = rng.uniform(0, 255);
    int g = rng.uniform(0, 255);
    int r = rng.uniform(0, 255);

    Vec3i cir = circles[i];
    circle(outputImage, Point(cir[0], cir[1]), cir[2], Scalar(b, g, r), 2, LINE_AA);
  }

  /*
  // // Draw the circles on the image.
  // for (size_t i = 0; i < circles.size(); i++)
  // {
  //   Vec3i c = circles[i];
  //   Point center = Point(c[0], c[1]);

  //   // circle outline
  //   int radius = c[2];
  //   circle(outputImage, center, radius, Scalar(0, 0, 255), 1, LINE_AA);
  // }
  */

  cout << "Circles Found: " << circles.size() << endl;

  // imshow("detected circles", outputImage);
  // waitKey();

#endif

  return 0;
}