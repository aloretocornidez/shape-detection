#include "homework-6.hpp"
// #include "erosion.hpp"
// #include "dilation.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{

  // Usage for the program
  if (argc != 2)
  {

    std::cout << "Usage: {programName} path/to/image" << std::endl;
    return -1;
 
  }

  // Read the image file
  cv::Mat inputImage = imread(argv[1], cv::IMREAD_GRAYSCALE);

  // Check for failure when opening the image.
  if (inputImage.empty())
  {

    std::cout << "Could not open or find the image" << std::endl;
    return -1;
 
  }

  Mat gray;
  // cvtColor(inputImage, gray, COLOR_BGR2GRAY);
  gray = inputImage.clone();

  // imshow("grayscale", gray);
  // waitKey();
  //
  medianBlur(gray, gray, 5);
  // imshow("blurred", gray);
  // waitKey();

  vector<Vec3f> circles;

  HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
               1, // change this value to detect circles with different distances to each other
               300, 37, 1, 200  // change the last two parameters
                               // (min_radius & max_radius) to detect larger circles
  );

  
  for (size_t i = 0; i < circles.size(); i++)
  {
    Vec3i c = circles[i];
    Point center = Point(c[0], c[1]);

    // circle center
    // circle(inputImage, center, 1, Scalar(255, 0, 0), 1, LINE_AA);


    // circle outline
    int radius = c[2];
    circle(inputImage, center, radius, Scalar(128,1, 1), 1, LINE_AA);
  }

  std::cout<< "Circles Found: " << circles.size() << std::endl;
  

  imshow("detected circles", inputImage);
  waitKey();

  return 0;
}
