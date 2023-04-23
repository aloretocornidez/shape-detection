#include "hough-transform.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

// Thread Block and Memory Allocation Parameters.
#define BLOCK_DIMENSIONS 32

// Hough Tranform Kernel Definitions
#define HOUGH_TRANSFORM_NAIVE_KERNEL 1
#define DEGREES_TO_RADIANS 0.0120830485

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
        maximumRadius = min(srcImage.rows - 1, srcImage.cols - 1) / 2;
    }

    /* Begin Algoritm */
    for (int radius = minimumRadius; radius < maximumRadius; radius++)
    {
        // Note, the threshold for the number of pixels is dynamically set within the loop.

        int threshold = ((log(radius * 2 / 3)) * 80) / log(3);
        // std::cout << "Testing Radius: " << radius << " | With Threshold: " << threshold << std::endl;

        for (int row = radius; row < srcImage.rows - radius; row += distance)
        {
            for (int column = radius; column < srcImage.cols - radius; column += distance)
            {
                // std::cout << "Testing (row, column): (" << row << ", " << column << ")" << std::endl;
                int accumulator = 0;
                // Check if the a circle exists at the coordinate point (with the current radius)
                for (int theta = 0; theta < 360; theta++)
                {
                    int x;
                    int y;
                    int deltaX = cos(theta * DEGREES_TO_RADIANS) * radius;
                    int deltaY = sin(theta * DEGREES_TO_RADIANS) * radius;

                    // Checking all 4 cardinal directions.
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
                    // std::cout << "Accumulator was greater than threshold. :(" << accumulator << ", " << threshold << ")" << std::endl;
                    // printf("Adding a set of circle parameters: [%d, %d, %d]\n\n", row, column, radius);

                    srcCircles.push_back({(float)row, (float)column, (float)radius});
                }
            }
        }
    }

    std::cout << "Execution of the Hough Transform on the CPU comleted." << std::endl;
}

// This kernel uses the global memory to write to the R-table.
// Each thread will test
__global__ void hough_transform_kernel_naive(uchar *srcImage, unsigned int *rTable, int imageRows, int imageColumns, int minimumRadius, int maximumRadius, int distance)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int column = threadIdx.x + blockIdx.x * blockDim.x;

    // DEBUG STATEMENTS
    // minimumRadius = 50;
    // maximumRadius = 55;

    // Check if the thread is within image bounds.
    if (row < (imageRows + minimumRadius) && column < (imageColumns + minimumRadius))
    {

        for (int radius = minimumRadius; radius < maximumRadius; radius++)
        {
            for (int theta = 0; theta < 360; theta++)
            {
                int deltaX = cos(DEGREES_TO_RADIANS * theta) * radius;
                int deltaY = sin(DEGREES_TO_RADIANS * theta) * radius;

                int imageIndex = (row + deltaY) * imageColumns + (row + deltaX);
                if (false)
                    srcImage[imageIndex] = theta % 255;
            }
        }
    }
}

void houghTransform(cv::Mat &srcImage, std::vector<cv::Vec3f> &circles, int method)
{
    /* Parameters Until we make them modular */
    int distance = 1;
    int minimumRadius = 18;  // 20 is the min for the test image.
    int maximumRadius = 100; // 100 is max radius for the test image.
    int imageRows = srcImage.rows;
    int imageColumns = srcImage.cols;

    // Run Hough Transform on the CPU and return.
    if (method == 0)
    {
        cpuKernelHoughTransform(srcImage, circles, distance, minimumRadius, maximumRadius);
        return;
    }

    // Allocating an r-table to populate parameters for each shape.
    unsigned int *rTable;
    rTable = (unsigned int *)malloc(imageColumns * imageRows * (maximumRadius - minimumRadius) * sizeof(unsigned int));
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
    err = cudaMemcpy(gpuImageBuffer, srcImage.ptr<uchar>(0, 0), imageRows * imageColumns * sizeof(uchar), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    // std::cout << "Cuda Memory Allocated." << std::endl;

    /*
     *
     * Execute the hough transform on the specified kernel, populating the accumulator table.
     *
     */
    if (method == HOUGH_TRANSFORM_NAIVE_KERNEL)
    {

        dim3 mygrid(ceil(imageColumns / (BLOCK_DIMENSIONS * 1.0)), ceil(imageRows / (BLOCK_DIMENSIONS * 1.0)));
        dim3 myblock(BLOCK_DIMENSIONS, BLOCK_DIMENSIONS);

        hough_transform_kernel_naive<<<mygrid, myblock>>>(gpuImageBuffer, rTable, imageRows, imageColumns, minimumRadius, maximumRadius, distance);
    }
    else
    {
        std::cout << "Invalid Kernel Method Chosen. | (method): " << method << std::endl;
    }

    // Copy data from device to host
    err = cudaMemcpy(srcImage.ptr<uchar>(0, 0), gpuImageBuffer, imageRows * imageColumns * sizeof(uchar), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cudaFree(gpuImageBuffer);
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Free allocated memory
    free(rTable);
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
