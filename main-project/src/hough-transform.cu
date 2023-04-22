#include "hough-transform.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

// Thread Block and Memory Allocation Parameters.
#define BLOCK_DIMENSIONS 32

// Hough Tranform Kernel Definitions
#define HOUGH_TRANSFORM_NAIVE_KERNEL 1

// Sample Kernel for memory access of an image.
#if 0
__global__ void set_image_to_value(uchar *inputImage, int height, int width)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int column = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < height && column < width)
    {

        inputImage[row * width + column] = 128;
    }
}
#endif

__global__ void hough_transform_kernel_naive(uchar *inputImage, int height, int width, cv::InputArray circles)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int column = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < height && column < width)
    {

        inputImage[row * width + column] = 128;
    }
}

void cpuKernelHoughTransform(cv::Mat &inputImage, cv::InputArray &circles, int minimumRadius, int maxRadius)
{
    std::cout << "Executing the Hough Transform on the CPU." << std::endl;

    if (minimumRadius > 1)
    {
        std::cerr << "Minimum radius must be 1 or greater." << std::endl;
        exit(-1);
    }

    /* Begin Algoritm */

    // Create Accumulator space (array to hold values of the x-coordinate, y-coordinate, and radius of the circle)
    

    // For each possible value of a, find each b that satisfies the equation (i - a)^2 + (j-b)^2 = r^2


    // Search for a local Maxima in the accumulator space.
    

    // Append the local maxima found to the circles array.








    for (int row = 0; row < inputImage.rows; row++)
    {
        for (int column = 0; column < inputImage.cols; column++)
        {
        }
    }
}

void houghTransform(cv::Mat &grayscaleInputImage, cv::InputArray &circles, int method)
{

    // Run Hough Transform on the CPU and return.
    if (method == 0)
    {
        cpuKernelHoughTransform(grayscaleInputImage, circles, 1, 1000);
        return;
    }

    int imageRows = grayscaleInputImage.rows;
    int imageColumns = grayscaleInputImage.cols;

    // Initialize pointer for the GPU memory
    uchar *gpuImageBuffer;

    // Allocate GPU Memory
    cudaError_t err = cudaMalloc((void **)&gpuImageBuffer, imageRows * imageColumns * sizeof(uchar));
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Copy Data from host to Device
    err = cudaMemcpy(gpuImageBuffer, grayscaleInputImage.ptr<uchar>(0, 0), imageRows * imageColumns * sizeof(uchar), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    // std::cout << "Cuda Memory Allocated." << std::endl;

    /*
     *
     * Execute the hough transform on the specified kernel.
     *
     */
    if (method == HOUGH_TRANSFORM_NAIVE_KERNEL)
    {

        dim3 mygrid(ceil(imageColumns / (BLOCK_DIMENSIONS * 1.0)), ceil(imageRows / (BLOCK_DIMENSIONS * 1.0)));
        dim3 myblock(BLOCK_DIMENSIONS, BLOCK_DIMENSIONS);

        hough_transform_kernel_naive<<<mygrid, myblock>>>(gpuImageBuffer, imageRows, imageColumns, circles);
    }
    else
    {
        std::cout << "Invalid Kernel Method Chosen. | (method): " << method << std::endl;
    }

    // Copy data from device to host
    err = cudaMemcpy(grayscaleInputImage.ptr<uchar>(0, 0), gpuImageBuffer, imageRows * imageColumns * sizeof(uchar), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cudaFree(gpuImageBuffer);
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Free allocated memory
    cudaFree(gpuImageBuffer);
}

__global__ void add_kernel_basic(int size, int *input1, int *input2)
{

    int thread = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread < size)
    {
        input1[thread] = input1[thread] + input2[thread];
    }
}

void cudaAddKernel(int array_size, int *array_1, int *array_2)
{

    std::cout << "Running Kernel Wrapper" << std::endl;

    // Initializing pointers to the gpu memory
    int *gpu_array_1;
    int *gput_array_2;

    // allocate memory on device, check for failure
    if (cudaMalloc((void **)&gpu_array_1, array_size * sizeof(int)) != cudaSuccess)
    {
        std::cout << "malloc error for gpuInput1" << std::endl;
    }
    if (cudaMalloc((void **)&gput_array_2, array_size * sizeof(int)) != cudaSuccess)
    {
        std::cout << "malloc error for gpuInput2" << std::endl;
    }

    // copy data to device, check for failure, free device if needed

    cudaError_t err; // Use this whenever calling cudaMalloc and cudaMemcpy.

    err = cudaMemcpy(gpu_array_1, array_1, array_size * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(gput_array_2, array_2, array_size * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Execute kernel.
    dim3 mygrid(ceil(array_size / 256.0));
    dim3 myblock(256);

    add_kernel_basic<<<mygrid, myblock>>>(array_size, gpu_array_1, gput_array_2);

    // copy data to host, check for failure, free device if needed
    if (cudaMemcpy(array_1, gpu_array_1, array_size * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        cudaFree(gpu_array_1);
        cudaFree(gput_array_2);
        printf("data transfer error from device to host on input1\n");
    }
    if (cudaMemcpy(array_2, gput_array_2, array_size * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        cudaFree(gpu_array_1);
        cudaFree(gput_array_2);
        printf("data transfer error from device to host on input2\n");
    }

    cudaFree(gpu_array_1);
    cudaFree(gput_array_2);
}
