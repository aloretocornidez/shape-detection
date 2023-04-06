#include <opencv2/opencv.hpp>
#include "convenience.hpp"

using namespace cv;

// void openWindow(String windowName, Mat *image);
// void convolution(Mat *input, Mat *output);

int main(int argc, char **argv)
{

  // Usage for the program
  if (argc != 2)
  {
    std::cout << "Usage: {programName} path/to/image" << std::endl;
    return -1;
  }

  try
  {

    // Read the image file
    Mat inputImage = imread(argv[1], IMREAD_GRAYSCALE);

    // Check for failure when opening the image.
    if (inputImage.empty())
    {
      std::cout << "Could not open or find the image" << std::endl;
      return -1;
    }

    // Creating an image buffer.
    Mat buffer = inputImage.clone();
    Mat output = inputImage.clone();

    // Do image processing.
    for (int row = 0; row < buffer.rows; row++)
    {
      for (int column = 0; column < buffer.cols; column++)
      {
        int pixelValue = buffer.at<Vec3b>(row, column)[0]; // blue
        // int value2 = buffer.at<Vec3b>(row, column)[1]; // green
        // int value3 = buffer.at<Vec3b>(row, column)[2]; // red

        if (pixelValue < 100)
        {
          output.at<Vec3b>(row, column)[0] = 255; // blue
        }
        else
        {
          output.at<Vec3b>(row, column)[0] = 0;
        }

        // std::cout << "(Row, Column): (" << row << "," << column << ")" << "\nPixel Value: " << pixelValue << std::endl;
      }
    }

    imwrite("test.png", output);

    Mat image = imread("test.png");
    openWindow("Threshold Address", &image);

    // Opening the image for viewing.
    String windowName = "Address"; // Name of the window
    openWindow("Address Original", &inputImage);
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << '\n';
  }

  return 0;
}

// void openWindow(String windowName, Mat *image)
// {

//   namedWindow(windowName); // Create a window

//   imshow(windowName, *image); // Show our image inside the created window.

//   waitKey(0); // Wait for any keystroke in the window

//   destroyWindow(windowName); // destroy the created window
// }
