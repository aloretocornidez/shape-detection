#include "hough-transform.hpp"
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <vector>

cudaError_t err;
float milliseconds = 0;

// Thread Block and Memory Allocation Parameters.
#define BLOCK_DIMENSIONS 32

// Hough Tranform Kernel Definitions
#define HOUGH_TRANSFORM_NAIVE_KERNEL 1
#define HOUGH_TRANSFORM_NAIVE_KERNEL2 2
#define HOUGH_TRANSFORM_SHARED_LOCAL_ACCUMULATOR 3
#define DEGREES_TO_RADIANS 0.0120830485

//Streaming implementation simulation parameters
//For our basic implementation, number of images should be a multiple of stream size
#define STREAM_SIZE 1
#define NUM_IMAGES 1

//For our Convolution Kernels masks
#define GAUSSIAN_KERNEL_SIZE 5
#define sigma 1.4
#define EDGE_DETECTION_SIZE 3
__constant__ float gaussianMask[GAUSSIAN_KERNEL_SIZE*GAUSSIAN_KERNEL_SIZE];
__constant__ float edgeMask1[EDGE_DETECTION_SIZE*EDGE_DETECTION_SIZE];
__constant__ float edgeMask2[EDGE_DETECTION_SIZE*EDGE_DETECTION_SIZE];

//Grayscale to RGB
#define CHANNELS 3 // we have 3 channels corresponding to RGB
// The input image is encoded as unsigned characters [0, 255]
__global__ void colorConvert(uchar * bgrImage,
    uchar * grayImage,
    int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width && y < height) {
        // get 1D coordinate for the grayscale image
        int grayOffset = y*width + x;
        // one can think of the RGB image havings
        // CHANNEL times columns than the gray scale image
        int rgbOffset = grayOffset*CHANNELS;
        uchar b = bgrImage[rgbOffset]; // blue value for pixel
        uchar g = bgrImage[rgbOffset + 1]; // green value for pixel
        uchar r = bgrImage[rgbOffset + 2]; // red value for pixel
        // perform the rescaling and store it
        // We multiply by floating point constants
        grayImage[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

//Gaussian Kernel convolution
#define convolutionBlockDim 32
__global__ void gaussianConvolution(uchar *inputImage, uchar* outputImage, int height, int width, int maskWidth){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    //Assuming we know the blocm width, to use 2D notation
    __shared__ uchar sharedM[convolutionBlockDim][convolutionBlockDim];
    //uchar pixVal = 0;

    int outputSizeSquare = convolutionBlockDim - (maskWidth - 1); //We will produce 28 pixel values for each 32 threads (per dimension)

    //Bring in the data
    int colIndexShared = blockIdx.x * outputSizeSquare - maskWidth / 2 + threadIdx.x;
    int rowIndexShared = blockIdx.y * outputSizeSquare - maskWidth / 2 + threadIdx.y;
    
    if(colIndexShared > -1 && colIndexShared < width && rowIndexShared > -1 && rowIndexShared < height){
        sharedM[threadIdx.y][threadIdx.x] = inputImage[rowIndexShared * width + colIndexShared];
    }
    else {
        sharedM[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    int colBorderCheck = blockIdx.x * outputSizeSquare  + threadIdx.x;
    int rowBorderCheck = blockIdx.y * outputSizeSquare  + threadIdx.y;

    //Boundary Checking
    if(rowBorderCheck < height && colBorderCheck < width){
        if(threadIdx.x < outputSizeSquare && threadIdx.y < outputSizeSquare){
            unsigned int pixVal = 0;
            for(int j = 0; j < maskWidth; j++){
                for(int k = 0; k < maskWidth; k++){
                    pixVal += sharedM[threadIdx.y + j][threadIdx.x + k] * gaussianMask[j * maskWidth + k];
                }
            }
            int colOutputIndex = blockIdx.x * outputSizeSquare + threadIdx.x;
            int rowOutputIndex = blockIdx.y * outputSizeSquare + threadIdx.y;
            if(pixVal > 255){
                pixVal = 255;
            }
            outputImage[rowOutputIndex * width + colOutputIndex] = pixVal;
        }
    }
}

//Edge filter 1 Kernel convolution
__global__ void edgeConvolution1(uchar *inputImage, uchar* outputImage, int height, int width, int maskWidth){
int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    //Assuming we know the blocm width, to use 2D notation
    __shared__ uchar sharedM[convolutionBlockDim][convolutionBlockDim];
    //uchar pixVal = 0;

    int outputSizeSquare = convolutionBlockDim - (maskWidth - 1); //We will produce 30 pixel values for each 32 threads (per dimension)

    //Bring in the data
    int colIndexShared = blockIdx.x * outputSizeSquare - maskWidth / 2 + threadIdx.x;
    int rowIndexShared = blockIdx.y * outputSizeSquare - maskWidth / 2 + threadIdx.y;
    
    if(colIndexShared > -1 && colIndexShared < width && rowIndexShared > -1 && rowIndexShared < height){
        sharedM[threadIdx.y][threadIdx.x] = inputImage[rowIndexShared * width + colIndexShared];
    }
    else {
        sharedM[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();
    
    int colBorderCheck = blockIdx.x * outputSizeSquare  + threadIdx.x;
    int rowBorderCheck = blockIdx.y * outputSizeSquare  + threadIdx.y;

    //Boundary Checking
    if(rowBorderCheck < height && colBorderCheck < width){
        if(threadIdx.x < outputSizeSquare && threadIdx.y < outputSizeSquare){
            unsigned int pixVal = 0;
            for(int j = 0; j < maskWidth; j++){
                for(int k = 0; k < maskWidth; k++){
                    pixVal += sharedM[threadIdx.y + j][threadIdx.x + k] * edgeMask1[j * maskWidth + k];
                }
            }

            // if(pixVal < 127){
            //     pixVal = 0;
            // }
            // else {
            //     pixVal = 255;
            // }
            if(pixVal > 255){
                pixVal = 255;
            }

            int colOutputIndex = blockIdx.x * outputSizeSquare + threadIdx.x;
            int rowOutputIndex = blockIdx.y * outputSizeSquare + threadIdx.y;

            outputImage[rowOutputIndex * width + colOutputIndex] = pixVal;
        }
    }
}

//Edge filter 2 Kernel convolution
__global__ void edgeConvolution2(uchar *inputImage, uchar* outputImage, int height, int width, int maskWidth){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    //Assuming we know the blocm width, to use 2D notation
    __shared__ uchar sharedM[convolutionBlockDim][convolutionBlockDim];
    //uchar pixVal = 0;

    int outputSizeSquare = convolutionBlockDim - (maskWidth - 1); //We will produce 30 pixel values for each 32 threads (per dimension)

    //Bring in the data
    int colIndexShared = blockIdx.x * outputSizeSquare - maskWidth / 2 + threadIdx.x;
    int rowIndexShared = blockIdx.y * outputSizeSquare - maskWidth / 2 + threadIdx.y;
    
    if(colIndexShared > -1 && colIndexShared < width && rowIndexShared > -1 && rowIndexShared < height){
        sharedM[threadIdx.y][threadIdx.x] = inputImage[rowIndexShared * width + colIndexShared];
    }
    else {
        sharedM[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();
    
    int colBorderCheck = blockIdx.x * outputSizeSquare  + threadIdx.x;
    int rowBorderCheck = blockIdx.y * outputSizeSquare  + threadIdx.y;

    //Boundary Checking
    if(rowBorderCheck < height && colBorderCheck < width){
        if(threadIdx.x < outputSizeSquare && threadIdx.y < outputSizeSquare){
            unsigned int pixVal = 0;
            for(int j = 0; j < maskWidth; j++){
                for(int k = 0; k < maskWidth; k++){
                    pixVal += sharedM[threadIdx.y + j][threadIdx.x + k] * edgeMask2[j * maskWidth + k];
                }
            }

            // if(pixVal < 127){
            //     pixVal = 0;
            // }
            // else {
            //     pixVal = 255;
            // }
            if(pixVal > 255){
                pixVal = 255;
            }

            int colOutputIndex = blockIdx.x * outputSizeSquare + threadIdx.x;
            int rowOutputIndex = blockIdx.y * outputSizeSquare + threadIdx.y;

            outputImage[rowOutputIndex * width + colOutputIndex] = pixVal;
        }
    }
}

//Combining operation for Sobel
__global__ void combineXYEdge(uchar *inputImageX, uchar *inputImageY, uchar* outputImage, int height, int width){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    int index = row * width + col;

    int intensity = 0;
    int threshold = 20 + 20;

    intensity = abs(inputImageX[index]) + abs(inputImageY[index]);

    //Binarize the image, cuda expects a black frame
    if(intensity > threshold){
        outputImage[index] = 0;
    }
    else {
        outputImage[index] = 255;
    }

}

// Sample Kernel for memory access of an image.
#if 0
__global__ void set_image_to_value(uchar *inputImage, int height, int width)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int column = threadIdx.x + blockIdx.x * blockDim.x;
>>>>>>> 7ff4737184a46b4f9dbc50590b894d56b93e0c05


void cpuKernelHoughTransform(cv::Mat &srcImage, std::vector<cv::Vec3f> &srcCircles, int distance, int minimumRadius, int maximumRadius)
{
    std::cout << "Executing the Hough Transform on the CPU." << std::endl;

    if (minimumRadius < 0)
    {
        std::cerr << "Minimum radius must be 1 or greater." << std::endl;
        exit(-1);
    }
    if (minimumRadius == 0)
    {
        minimumRadius = 5;
    }
    if (maximumRadius == 0)
    {
        maximumRadius = std::min(srcImage.rows - 1, srcImage.cols - 1) / 2;
    }

    /* Begin Algoritm */
    for (int radius = minimumRadius; radius < maximumRadius; radius++)
    {
        // Note, the threshold for the number of pixels is dynamically set.
        int threshold = ((log(radius * 2 / 3)) * 80) / log(3);

        for (int row = radius; row < srcImage.rows - radius; row += distance)
        {
            for (int column = radius; column < srcImage.cols - radius; column += distance)
            {
                int accumulator = 0;

                // Check if the a circle exists at the coordinate point (with the current radius).
                for (int theta = 0; theta < 360; theta++)
                {
                    // Checking all 4 cardinal directions.
                    int x;
                    int y;
                    int deltaX = cos(theta * DEGREES_TO_RADIANS) * radius;
                    int deltaY = sin(theta * DEGREES_TO_RADIANS) * radius;
                    x = deltaX + column;
                    y = deltaY + row;

                    if (srcImage.at<uchar>(x, y) < 10)
                    {
                        accumulator++;
                    }
                }

                // Adding the coordinate if the contained enough edge pixels.
                if (accumulator > threshold)
                {
                    srcCircles.push_back({(float)row, (float)column, (float)radius});
                }
            }
        }
    }

    std::cout << "Execution of the Hough Transform on the CPU completed." << std::endl;
}

// This kernel uses the global memory to write to the R-table.
// Each thread will test
__global__ void hough_transform_kernel_naive(uchar *srcImage, unsigned int *rTable, int imageRows, int imageColumns, int minimumRadius, int maximumRadius, int distance)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int column = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if the thread is within image bounds.
    if (row < (imageRows - minimumRadius) && row > minimumRadius && column < (imageColumns - minimumRadius) && column > minimumRadius)
    {

        for (int radius = minimumRadius; radius < maximumRadius - minimumRadius; radius++)
        {
            int threshold = ((log10f(radius * 2 / 3)) * 80) / log10f(3);
            //printf("Threshold: %d\n", threshold);

            for (int theta = 0; theta < 360; theta++)
            {
                int deltaX = cos(DEGREES_TO_RADIANS * theta) * radius;
                int deltaY = sin(DEGREES_TO_RADIANS * theta) * radius;

                int imageIndex = (row + deltaY) * imageColumns + (column + deltaX);

                int pixelValue = srcImage[imageIndex];
                if (pixelValue < 127)
                {
                    atomicAdd(&rTable[(radius * imageColumns * imageRows) + row * imageColumns + column], 1);
                }
            }
        }
    }
}

__global__ void hough_transform_kernel_naive2(uchar *srcImage, unsigned int *rTable, int imageRows, int imageColumns, int minimumRadius, int maximumRadius, int distance)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int column = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if the thread is within image bounds.
    if (row < (imageRows - minimumRadius) && row > minimumRadius && column < (imageColumns - minimumRadius) && column > minimumRadius)
    {

        for (int radius = minimumRadius; radius < maximumRadius - minimumRadius; radius++)
        {
            int threshold = ((log10f(radius * 2 / 3)) * 80) / log10f(3);
            int accumulator = 0;

            for (int theta = 0; theta < 360; theta++)
            {
                int deltaX = cos(DEGREES_TO_RADIANS * theta) * radius;
                int deltaY = sin(DEGREES_TO_RADIANS * theta) * radius;

                int imageIndex = (row + deltaY) * imageColumns + (column + deltaX);

                int pixelValue = srcImage[imageIndex];
                if (pixelValue < 127)
                {

                    accumulator++;
                }
            }

            atomicAdd(&rTable[(radius * imageColumns * imageRows) + row * imageColumns + column], accumulator);
        }
    }
}

// Did not take into account the amount of memory available on the system.
__global__ void hough_transform_kernel_shared_local_accumulator(uchar *srcImage, unsigned int *rTable, int imageRows, int imageColumns, int minimumRadius, int maximumRadius, int distance)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int column = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ uchar *sharedSrcImage;

    uchar *sharedSrcImageData = sharedSrcImage;

    if (row < imageRows && column < imageColumns)
    {
        sharedSrcImage[row * imageColumns + column] = srcImage[row * imageColumns + column];
    }

    __syncthreads();

    // Check if the thread is within image bounds.
    if (row < (imageRows - minimumRadius) && row > minimumRadius && column < (imageColumns - minimumRadius) && column > minimumRadius)
    {

        for (int radius = minimumRadius; radius < maximumRadius - minimumRadius; radius++)
        {
            int threshold = ((log10f(radius * 2 / 3)) * 80) / log10f(3);
            int accumulator = 0;

            for (int theta = 0; theta < 360; theta++)
            {
                int deltaX = cos(DEGREES_TO_RADIANS * theta) * radius;
                int deltaY = sin(DEGREES_TO_RADIANS * theta) * radius;

                int imageIndex = (row + deltaY) * imageColumns + (column + deltaX);

                // int pixelValue = srcImage[imageIndex];
                int pixelValue = sharedSrcImageData[imageIndex];

                if (pixelValue < 10)
                {

                    accumulator++;
                }
            }

            atomicAdd(&rTable[(radius * imageColumns * imageRows) + row * imageColumns + column], accumulator);
        }
    }
}


// Change the algorithm so that each thread focuses on one part of the image.
<<<<<<< Updated upstream
__global__ void hough_transform_kernel_shared_local_accumulator2(uchar *srcImage, unsigned int *rTable, int imageRows, int imageColumns, int minimumRadius, int maximumRadius, int distance)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int column = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ uchar *sharedSrcImage;

    uchar *sharedSrcImageData = sharedSrcImage;

    if (row < imageRows && column < imageColumns)
    {
        sharedSrcImage[row * imageColumns + column] = srcImage[row * imageColumns + column];
    }

    __syncthreads();

    // Check if the thread is within image bounds.
    if (row < (imageRows - minimumRadius) && row > minimumRadius && column < (imageColumns - minimumRadius) && column > minimumRadius)
    {

        for (int radius = minimumRadius; radius < maximumRadius - minimumRadius; radius++)
        {
            int threshold = ((log10f(radius * 2 / 3)) * 80) / log10f(3);
            int accumulator = 0;

            for (int theta = 0; theta < 360; theta++)
            {
                int deltaX = cos(DEGREES_TO_RADIANS * theta) * radius;
                int deltaY = sin(DEGREES_TO_RADIANS * theta) * radius;

                int imageIndex = (row + deltaY) * imageColumns + (column + deltaX);

                // int pixelValue = srcImage[imageIndex];
                int pixelValue = sharedSrcImageData[imageIndex];

                if (pixelValue < 10)
                {

                    accumulator++;
                }
            }

            atomicAdd(&rTable[(radius * imageColumns * imageRows) + row * imageColumns + column], accumulator);
        }
    }
}
=======
// __global__ void hough_transform_kernel_shared_local_accumulator(uchar *srcImage, unsigned int *rTable, int imageRows, int imageColumns, int minimumRadius, int maximumRadius, int distance)
// {
//     int row = threadIdx.y + blockIdx.y * blockDim.y;
//     int column = threadIdx.x + blockIdx.x * blockDim.x;
//
//     extern __shared__ uchar *sharedSrcImage;
//
//     uchar *sharedSrcImageData = sharedSrcImage;
//
//     if (row < imageRows && column < imageColumns)
//     {
//         sharedSrcImage[row * imageColumns + column] = srcImage[row * imageColumns + column];
//     }
//
//     __syncthreads();
//
//     // Check if the thread is within image bounds.
//     if (row < (imageRows - minimumRadius) && row > minimumRadius && column < (imageColumns - minimumRadius) && column > minimumRadius)
//     {
//
//         for (int radius = minimumRadius; radius < maximumRadius - minimumRadius; radius++)
//         {
//             int threshold = ((log10f(radius * 2 / 3)) * 80) / log10f(3);
//             int accumulator = 0;
//
//             for (int theta = 0; theta < 360; theta++)
//             {
//                 int deltaX = cos(DEGREES_TO_RADIANS * theta) * radius;
//                 int deltaY = sin(DEGREES_TO_RADIANS * theta) * radius;
//
//                 int imageIndex = (row + deltaY) * imageColumns + (column + deltaX);
//
//                 // int pixelValue = srcImage[imageIndex];
//                 int pixelValue = sharedSrcImageData[imageIndex];
//
//                 if (pixelValue < 10)
//                 {
//
//                     accumulator++;
//                 }
//             }
//
//             atomicAdd(&rTable[(radius * imageColumns * imageRows) + row * imageColumns + column], accumulator);
//         }
//     }
// }
>>>>>>> Stashed changes

void parseRTable(std::vector<cv::Vec3f> &circles, unsigned int *rTable, int minimumRadius, int maximumRadius, int imageRows, int imageColumns)
{

    for (int row = minimumRadius; row < imageRows - minimumRadius; row++)
    {
        for (int column = minimumRadius; column < imageColumns - minimumRadius; column++)
        {
            for (int radius = 0; radius < (maximumRadius - minimumRadius); radius++)
            {
                // Check if the image at that coordinate is greater than the threshold.
                // If so, then append the circle to the circles vector.
                int rValue = rTable[(radius * imageColumns * imageRows) + row * imageColumns + column];
                if (rValue > 325)
                {

                    // printf("Adding Circle at (row, column, radius) | (%d, %d, %d) | RValue: %d\n", row, column, radius, rValue);

                    circles.push_back({(float)(column), (float)(row), (float)(radius)});

                    // circles.push_back({(float)(row + minimumRadius), (float)(column + minimumRadius), (float)(radius + minimumRadius)});
                }
            }
        }
    }
}

void houghTransform(cv::Mat &srcImage, std::vector<cv::Vec3f> &circles, int method)
{
    /* Parameters Until we make them modular */
    int distance = 1;
    int minimumRadius = 18;  // 20 is the min for the test image.
    int maximumRadius = 150; // 100 is max radius for the test image.
    int imageRows = srcImage.rows;
    int imageColumns = srcImage.cols;
    //printf("Channels: %d\n", srcImage.channels());
    cv::imwrite("testImage3.jpg", srcImage);


    // Run Hough Transform on the CPU and return.
    if (method == 0)
    {

        cv::Mat grayscale;
        cv::cvtColor(srcImage, grayscale, cv::COLOR_BGR2GRAY);


        cudaEvent_t cpuStart, cpuStop;
        cudaEventCreate(&cpuStart);
        cudaEventCreate(&cpuStop);

        cudaEventRecord(cpuStart);

        std::cout << "Executing Hough Transform on the CPU" << std::endl;
        cpuKernelHoughTransform(grayscale, circles, distance, minimumRadius, maximumRadius);

        cudaEventRecord(cpuStop);
        cudaEventSynchronize(cpuStop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, cpuStart, cpuStop);

        std::cout << "Execution Time on the CPU: " << milliseconds << std::endl;
        return;
    }
    
    //Populate three important kernels
    float gaussianFilter[GAUSSIAN_KERNEL_SIZE][GAUSSIAN_KERNEL_SIZE];
    //Initialize Gaussian Filter
    // initialising standard deviation to 1.0
    float r;
    float s = 2.0 * sigma * sigma;
    // sum is for normalization
    float sum = 0.0;
    // generating 5x5 kernel
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            r = sqrt(x * x + y * y);
            gaussianFilter[x + 2][y + 2] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += gaussianFilter[x + 2][y + 2];
        }
    }
    // normalising the Kernel
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            gaussianFilter[i][j] /= sum;

    //Copying it to 1d array
    float* gaussian1D = (float*)malloc(GAUSSIAN_KERNEL_SIZE*GAUSSIAN_KERNEL_SIZE*sizeof(float));
    for(int i = 0; i < GAUSSIAN_KERNEL_SIZE; i++){
        for(int j = 0; j < GAUSSIAN_KERNEL_SIZE; j++){
            gaussian1D[i * GAUSSIAN_KERNEL_SIZE + j] = gaussianFilter[i][j];
        }
    }
    std::cout << "Gaussian filter generated\n";

    cudaMemcpyToSymbol(gaussianMask, gaussian1D, GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE * sizeof(float));

    float sobelX[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    cudaMemcpyToSymbol(edgeMask1, sobelX, EDGE_DETECTION_SIZE * EDGE_DETECTION_SIZE * sizeof(float));


    float sobelY[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    cudaMemcpyToSymbol(edgeMask2, sobelY, EDGE_DETECTION_SIZE * EDGE_DETECTION_SIZE * sizeof(float));
    std::cout << "Sobel values generated\n";



    //Allocate memory for all arrays and intermediate results
    //Memory Declarations and Allocations
    uchar* hostBGRInput[NUM_IMAGES];
    uchar* deviceBGRInput[STREAM_SIZE];
    uchar* hostGrayImage[NUM_IMAGES];
    uchar* deviceGrayImage[STREAM_SIZE];
    uchar* hostConvolved1[NUM_IMAGES];
    uchar* deviceConvolved1[STREAM_SIZE];
    uchar* hostConvolved2[NUM_IMAGES];
    uchar* deviceConvolved2[STREAM_SIZE];
    uchar* hostConvolved3[NUM_IMAGES];
    uchar* deviceConvolved3[STREAM_SIZE]; 
    uchar* hostBinaryImage[NUM_IMAGES]; 
    uchar* deviceBinaryImage[STREAM_SIZE]; 
    // Allocating an r-table to populate parameters for each shape.
    // Allocate the R table on the GPU.
    unsigned int *deviceRTable[STREAM_SIZE];
    unsigned int *hostRTable[NUM_IMAGES];
    

    //Malloc Host
    for (int i = 0; i < NUM_IMAGES; i++){
        cudaHostAlloc( (void**)&hostBGRInput[i], imageRows * imageColumns * 3 * sizeof(uchar), cudaHostAllocDefault);
        cudaHostAlloc( (void**)&hostGrayImage[i], imageRows * imageColumns * sizeof(uchar), cudaHostAllocDefault);
        cudaHostAlloc( (void**)&hostConvolved1[i], imageRows * imageColumns * sizeof(uchar), cudaHostAllocDefault);
        cudaHostAlloc( (void**)&hostConvolved2[i], imageRows * imageColumns * sizeof(uchar), cudaHostAllocDefault);
        cudaHostAlloc( (void**)&hostConvolved3[i], imageRows * imageColumns * sizeof(uchar), cudaHostAllocDefault);
        cudaHostAlloc( (void**)&hostBinaryImage[i], imageRows * imageColumns * sizeof(uchar), cudaHostAllocDefault);
        cudaHostAlloc( (void**)&hostRTable[i], imageColumns * imageRows * (maximumRadius - minimumRadius) * sizeof(unsigned int), cudaHostAllocDefault);

        // hostBGRInput[i] = (uchar*)malloc(rows * columns * CHANNELS * sizeof(uchar));
        // hostGrayImage[i] = (uchar*)malloc(rows * columns * sizeof(uchar));
        // hostConvolved1[i] = (uchar*)malloc(rows * columns * sizeof(uchar));
        // hostConvolved2[i] = (uchar*)malloc(rows * columns * sizeof(uchar));
        // hostConvolved3[i] = (uchar*)malloc(rows * columns  * sizeof(uchar));
    }

    //CudaMalloc
    for(int i = 0; i < STREAM_SIZE; i++){
        err = cudaMalloc((void**)&deviceBGRInput[i], imageRows * imageColumns * 3 * sizeof(uchar));
        err = cudaMalloc((void**)&deviceGrayImage[i], imageRows * imageColumns * sizeof(uchar));
        err = cudaMalloc((void**)&deviceConvolved1[i], imageRows * imageColumns * sizeof(uchar));
        err = cudaMalloc((void**)&deviceConvolved2[i], imageRows * imageColumns * sizeof(uchar));
        err = cudaMalloc((void**)&deviceConvolved3[i], imageRows * imageColumns * sizeof(uchar));
        err = cudaMalloc((void**)&deviceBinaryImage[i], imageRows * imageColumns * sizeof(uchar));

        err = cudaMalloc((void **)&deviceRTable[i], imageColumns * imageRows * (maximumRadius - minimumRadius) * sizeof(unsigned int));
        cudaMemset(deviceRTable[i], 0, imageColumns * imageRows * (maximumRadius - minimumRadius) * sizeof(unsigned int));
    }
    cudaDeviceSynchronize();
    std::cout << "Initial memory allocated\n";

    // //Generate input array on Host
    // std::vector<uchar> array;
    // if (srcImage.isContinuous()){
    //     // array.assign(mat.datastart, mat.dataend); // <- has problems for sub-matrix like mat = big_mat.row(i)
    //     array.assign(srcImage.data, srcImage.data + srcImage.total()*srcImage.channels());
    //     printf("Image is continuous\n");
    // }
    //For our testing purposes, we will have copies of the same image
    uchar* pixelPtr = srcImage.ptr<uchar>(0, 0);
    int channels = srcImage.channels();
    //

    // for(int i = 0; i < foo.rows; i++)
    // {
    //     for(int j = 0; j < foo.cols; j++)
    //     {
    //         bgrPixel.val[0] = pixelPtr[i*foo.cols*cn + j*cn + 0]; // B
    //         bgrPixel.val[1] = pixelPtr[i*foo.cols*cn + j*cn + 1]; // G
    //         bgrPixel.val[2] = pixelPtr[i*foo.cols*cn + j*cn + 2]; // R
            
    //         // do something with BGR values...
    //     }
    // }

    //Initialize Input Image
    //Debugging
    uchar testArray[270][330][3];
    for(int s = 0; s < NUM_IMAGES; s++){
        for(int i = 0; i < imageRows; i++){
            for(int j = 0; j < imageColumns; j++){
                for(int k = 0; k < channels; k++){
                    //std::cout << "Here1\n";
                    int index = (i * imageColumns * channels) + (j * channels) + k;
                    hostBGRInput[s][index] = pixelPtr[index];
                    //std::cout << "Here2\n";
                }
            }
        }
    }

    //Ensure host bgr is correct
    cv::Mat image = cv::Mat(imageRows, imageColumns, CV_8UC3);
    image.data = hostBGRInput[0];
    cv::imwrite("testImage2.jpg", image);
    printf("Rows: %d, Cols: %d\n", imageRows, imageColumns);
    



    //Dimensions for blocks and grayscale grid
    int blockSize = 32;
    //int maskVal = 5;
    int gaussianTile = 28;
    int edgeTile = 30;
    dim3 blockDim(blockSize,blockSize), gridDim( 1 + (imageColumns - 1) / blockSize,
                                             1 + (imageRows - 1) / blockSize);
    //Grid for Gaussian Convolution
    dim3 gaussianGridDim(ceil((float) (imageColumns) / gaussianTile),
                ceil((float) (imageRows) / gaussianTile));

    dim3 edgeGridDim(ceil((float) (imageColumns) / edgeTile),
                ceil((float) (imageRows) / edgeTile));

    //Stream Initialization
    //Initialize stream
    cudaStream_t streams[STREAM_SIZE];
    for(int i = 0; i < STREAM_SIZE; i++){
        cudaStreamCreate(&streams[i]);
    }

    //BEGIN OPERATIONS
    //Copy Input array to GPU
    cudaEvent_t kernelLaunch, kernelEnd;
    cudaEventCreate(&kernelLaunch);
    cudaEventCreate(&kernelEnd);
    cudaEventRecord(kernelLaunch);
    std::cout << "Here\n";

    for(int j = 0; j < NUM_IMAGES; j = j + STREAM_SIZE){
        for(int i = 0; i < STREAM_SIZE; i++){
            err = cudaMemcpyAsync(deviceBGRInput[i], hostBGRInput[i + j], imageRows * imageColumns * 3 * sizeof(uchar), cudaMemcpyHostToDevice, streams[i]);
            
            colorConvert<<<gridDim, blockDim, 0, streams[i]>>>(deviceBGRInput[i], deviceGrayImage[i], imageColumns, imageRows);
            err = cudaMemcpyAsync(hostGrayImage[i + j], deviceGrayImage[i], imageRows * imageColumns * sizeof(uchar), cudaMemcpyDeviceToHost, streams[i]);

            gaussianConvolution<<<gaussianGridDim, blockDim, 0, streams[i]>>>
                (deviceGrayImage[i], deviceConvolved1[i], imageRows, imageColumns, GAUSSIAN_KERNEL_SIZE);
            err = cudaMemcpyAsync(hostConvolved1[i + j], deviceConvolved1[i], imageRows * imageColumns * sizeof(uchar), cudaMemcpyDeviceToHost, streams[i]);

            
            edgeConvolution1<<<edgeGridDim, blockDim, 0, streams[i]>>>
                (deviceConvolved1[i], deviceConvolved2[i], imageRows, imageColumns, EDGE_DETECTION_SIZE);
            err = cudaMemcpyAsync(hostConvolved2[i + j], deviceConvolved2[i], imageRows * imageColumns * sizeof(uchar), cudaMemcpyDeviceToHost, streams[i]);

            
            edgeConvolution2<<<edgeGridDim, blockDim, 0, streams[i]>>>
                (deviceConvolved1[i], deviceConvolved3[i], imageRows, imageColumns, EDGE_DETECTION_SIZE);
            err = cudaMemcpyAsync(hostConvolved3[i + j], deviceConvolved3[i], imageRows * imageColumns * sizeof(uchar), cudaMemcpyDeviceToHost, streams[i]);

            
            combineXYEdge<<<gaussianGridDim, blockDim, 0, streams[i]>>>
                (deviceConvolved2[i], deviceConvolved3[i], deviceBinaryImage[i], imageRows, imageColumns);
            err = cudaMemcpyAsync(hostBinaryImage[i + j], deviceBinaryImage[i], imageRows * imageColumns * sizeof(uchar), cudaMemcpyDeviceToHost, streams[i]);

                
            if (method == HOUGH_TRANSFORM_NAIVE_KERNEL)
                {
                    // cudaEvent_t startNaiveHoughTransform, stopNaiveHoughTransform;
                    // cudaEventCreate(&startNaiveHoughTransform);
                    // cudaEventCreate(&stopNaiveHoughTransform);

                    // cudaEventRecord(startNaiveHoughTransform);

                    std::cout << "Executing Hough Transform on the Naive Kernel" << std::endl;

                    dim3 mygrid(ceil(imageColumns / (BLOCK_DIMENSIONS * 1.0)), ceil(imageRows / (BLOCK_DIMENSIONS * 1.0)));
                    dim3 myblock(BLOCK_DIMENSIONS, BLOCK_DIMENSIONS);

                    hough_transform_kernel_naive<<<mygrid, myblock, 0, streams[i]>>>(deviceBinaryImage[i], deviceRTable[i], imageRows, imageColumns, minimumRadius, maximumRadius, distance);

                    // std::cout << "Hough Transform Naive Kernel Execution Complete" << std::endl;
                    // cudaEventRecord(stopNaiveHoughTransform);
                    // cudaEventSynchronize(stopNaiveHoughTransform);
                    // float milliseconds = 0;
                    // cudaEventElapsedTime(&milliseconds, startNaiveHoughTransform, stopNaiveHoughTransform);

                    //std::cout << "Execution Time on the GPU: " << milliseconds << std::endl;
                }
                else if (method == HOUGH_TRANSFORM_NAIVE_KERNEL2)
                {
                    // std::cout << "Executing Hough Transform on the Local Accumulator Kernel 2" << std::endl;
                    // cudaEvent_t startNaiveHoughTransform, stopNaiveHoughTransform;
                    // cudaEventCreate(&startNaiveHoughTransform);
                    // cudaEventCreate(&stopNaiveHoughTransform);

                    // cudaEventRecord(startNaiveHoughTransform);

                    dim3 mygrid(ceil(imageColumns / (BLOCK_DIMENSIONS * 1.0)), ceil(imageRows / (BLOCK_DIMENSIONS * 1.0)));
                    dim3 myblock(BLOCK_DIMENSIONS, BLOCK_DIMENSIONS);

                    hough_transform_kernel_naive<<<mygrid, myblock>>>(deviceBinaryImage[i], deviceRTable[i], imageRows, imageColumns, minimumRadius, maximumRadius, distance);

                    // std::cout << "Hough Transform Naive Kernel Execution Complete" << std::endl;
                    // cudaEventRecord(stopNaiveHoughTransform);
                    // cudaEventSynchronize(stopNaiveHoughTransform);
                    // float milliseconds = 0;
                    // cudaEventElapsedTime(&milliseconds, startNaiveHoughTransform, stopNaiveHoughTransform);

                    // std::cout << "Execution Time on the GPU: " << milliseconds << std::endl;
                }       

            err = cudaMemcpyAsync(hostRTable[i + j], deviceRTable[i], imageColumns * imageRows * (maximumRadius - minimumRadius) * sizeof(unsigned int), cudaMemcpyDeviceToHost, streams[i]);
            //cudaStreamSynchronize(streams[i]);
            //err = cudaMemcpyAsync(hostConvolved3[i + j], deviceConvolved3[i + j], imageRows * imageColumns * sizeof(uchar), cudaMemcpyDeviceToHost, streams[i]);
            //
        }
    }
    cudaDeviceSynchronize();
    cudaEventRecord(kernelEnd);
    cudaEventSynchronize(kernelEnd);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, kernelLaunch, kernelEnd);
    printf("Total %d Images time: %fms\n", NUM_IMAGES, milliseconds);
    printf("Streams used: %d\n", STREAM_SIZE);
    printf("estimated time for 30 images: %fms\n", milliseconds * (30.0 / NUM_IMAGES));

    cv::Mat grayTest = cv::Mat(imageRows, imageColumns, CV_8UC1);
    grayTest.data = hostGrayImage[0];
    cv::imwrite("testImage4.jpg", grayTest);
    //printf("Rows: %d, Cols: %d\n", imageRows, imageColumns); 
    
    cv::Mat gaussianTest = cv::Mat(imageRows, imageColumns, CV_8UC1);
    gaussianTest.data = hostConvolved1[0];
    cv::imwrite("testImage5.jpg", gaussianTest);
    //printf("Rows: %d, Cols: %d\n", imageRows, imageColumns); 
    
    cv::Mat xConvTest = cv::Mat(imageRows, imageColumns, CV_8UC1);
    xConvTest.data = hostConvolved2[0];
    cv::imwrite("testImage6.jpg", xConvTest);
    //printf("Rows: %d, Cols: %d\n", imageRows, imageColumns); 

    cv::Mat yConvTest = cv::Mat(imageRows, imageColumns, CV_8UC1);
    yConvTest.data = hostConvolved3[0];
    cv::imwrite("testImage7.jpg", yConvTest);
    //printf("Rows: %d, Cols: %d\n", imageRows, imageColumns); 

    cv::Mat binaryTest = cv::Mat(imageRows, imageColumns, CV_8UC1);
    binaryTest.data = hostBinaryImage[0];
    cv::imwrite("testImage8.jpg", binaryTest);
    //printf("Rows: %d, Cols: %d\n", imageRows, imageColumns); 
    

    // // Allocate GPU Memory Initialize pointer for the GPU memory
    // uchar *gpuImageBuffer;
    // cudaError_t err = cudaMalloc((void **)&gpuImageBuffer, imageRows * imageColumns * sizeof(uchar));
    // if (err != cudaSuccess)
    // {
    //     printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    //     exit(EXIT_FAILURE);
    // }

    

    /*
     *
     * Execute the hough transform on the specified kernel, populating the accumulator table.
     *
     */
    
    // else if (method == HOUGH_TRANSFORM_SHARED_LOCAL_ACCUMULATOR)
    // {
    //     std::cout << "Executing Hough Transform on the Shared Memory Kernel." << std::endl;
    //     if (sizeof(uchar) * imageRows * imageColumns > 64000 / 4.0)
    //     {
    //         printf("The image is too large to run using shared memory. Running on global memory kernel. Rows: %d, Columns: %d\n", imageRows, imageColumns);
    //         exit(-1);
    //     }
    //     cudaEvent_t startNaiveHoughTransform, stopNaiveHoughTransform;
    //     cudaEventCreate(&startNaiveHoughTransform);
    //     cudaEventCreate(&stopNaiveHoughTransform);

    //     cudaEventRecord(startNaiveHoughTransform);

    //     dim3 mygrid(ceil(imageColumns / (BLOCK_DIMENSIONS * 1.0)), ceil(imageRows / (BLOCK_DIMENSIONS * 1.0)));
    //     dim3 myblock(BLOCK_DIMENSIONS, BLOCK_DIMENSIONS);

    //     hough_transform_kernel_naive<<<mygrid, myblock, sizeof(uchar) * imageRows * imageColumns>>>(gpuImageBuffer, deviceRTable, imageRows, imageColumns, minimumRadius, maximumRadius, distance);

    //     // std::cout << "Hough Transform Naive Kernel Execution Complete" << std::endl;
    //     cudaEventRecord(stopNaiveHoughTransform);
    //     cudaEventSynchronize(stopNaiveHoughTransform);
    //     float milliseconds = 0;
    //     cudaEventElapsedTime(&milliseconds, startNaiveHoughTransform, stopNaiveHoughTransform);

    //     std::cout << "Execution Time on the GPU: " << milliseconds << std::endl;
    // }
    // else
    // {
    //     std::cout << "Invalid Kernel Method Chosen. | (method): " << method << std::endl;
    //}

    /* Modications are not being made to the image, so no copy back to the host is required. */
    // Copy data from device to host
    
    // if (err != cudaSuccess)
    // {
    //     cudaFree(gpuImageBuffer);
    //     printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    //     exit(EXIT_FAILURE);
    // }

    parseRTable(circles, hostRTable[0], minimumRadius, maximumRadius, imageRows, imageColumns);

    // Free allocated memory
    cudaFree(deviceRTable);
    //cudaFree(gpuImageBuffer);
    //free(hostRTable[0]);
    std::cout << "Cuda Memory Freed" << std::endl;

    std::cout << "GPU Hough Transform Execution Complete" << std::endl;
}

