#include "homework-6.hpp"

void showImage(cv::String name, cv::Mat input)
{
    using namespace cv;
    imshow(name, input);


    waitKey(0);

    destroyAllWindows();
}
