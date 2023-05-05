#include "hough-transform.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Thread Block and Memory Allocation Parameters.
#define BLOCK_DIMENSIONS 32

// Hough Tranform Kernel Definitions
#define HOUGH_TRANSFORM_NAIVE_KERNEL 1
#define HOUGH_TRANSFORM_NAIVE_KERNEL2 2
#define HOUGH_TRANSFORM_SHARED_LOCAL_ACCUMULATOR 3
#define DEGREES_TO_RADIANS 0.0120830485



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

            for (int theta = 0; theta < 360; theta++)
            {
                int deltaX = cos(DEGREES_TO_RADIANS * theta) * radius;
                int deltaY = sin(DEGREES_TO_RADIANS * theta) * radius;

                int imageIndex = (row + deltaY) * imageColumns + (column + deltaX);

                int pixelValue = srcImage[imageIndex];
                if (pixelValue < 10)
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
                if (pixelValue < 10)
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
    int maximumRadius = 120; // 100 is max radius for the test image.
    int imageRows = srcImage.rows;
    int imageColumns = srcImage.cols;

    // Run Hough Transform on the CPU and return.
    if (method == 0)
    {

        cudaEvent_t cpuStart, cpuStop;
        cudaEventCreate(&cpuStart);
        cudaEventCreate(&cpuStop);

        cudaEventRecord(cpuStart);

        std::cout << "Executing Hough Transform on the CPU" << std::endl;
        cpuKernelHoughTransform(srcImage, circles, distance, minimumRadius, maximumRadius);

        cudaEventRecord(cpuStop);
        cudaEventSynchronize(cpuStop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, cpuStart, cpuStop);

        std::cout << "Execution Time on the CPU: " << milliseconds << std::endl;
        return;
    }

    // Allocate GPU Memory Initialize pointer for the GPU memory
    uchar *gpuImageBuffer;
    cudaError_t err = cudaMalloc((void **)&gpuImageBuffer, imageRows * imageColumns * sizeof(uchar));
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Allocating an r-table to populate parameters for each shape.
    // Allocate the R table on the GPU.
    unsigned int *deviceRTable;
    err = cudaMalloc((void **)&deviceRTable, imageColumns * imageRows * (maximumRadius - minimumRadius) * sizeof(unsigned int));
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaMemset(deviceRTable, 0, imageColumns * imageRows * (maximumRadius - minimumRadius) * sizeof(unsigned int));
    // Copy Data from host to Device
    err = cudaMemcpy(gpuImageBuffer, srcImage.ptr<uchar>(0, 0), imageRows * imageColumns * sizeof(uchar), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    std::cout << "Cuda Memory Allocated." << std::endl;

    /*
     *
     * Execute the hough transform on the specified kernel, populating the accumulator table.
     *
     */
    if (method == HOUGH_TRANSFORM_NAIVE_KERNEL)
    {
        cudaEvent_t startNaiveHoughTransform, stopNaiveHoughTransform;
        cudaEventCreate(&startNaiveHoughTransform);
        cudaEventCreate(&stopNaiveHoughTransform);

        cudaEventRecord(startNaiveHoughTransform);

        // std::cout << "Executing Hough Transform on the Naive Kernel" << std::endl;

        dim3 mygrid(ceil(imageColumns / (BLOCK_DIMENSIONS * 1.0)), ceil(imageRows / (BLOCK_DIMENSIONS * 1.0)));
        dim3 myblock(BLOCK_DIMENSIONS, BLOCK_DIMENSIONS);

        hough_transform_kernel_naive<<<mygrid, myblock>>>(gpuImageBuffer, deviceRTable, imageRows, imageColumns, minimumRadius, maximumRadius, distance);

        // std::cout << "Hough Transform Naive Kernel Execution Complete" << std::endl;
        cudaEventRecord(stopNaiveHoughTransform);
        cudaEventSynchronize(stopNaiveHoughTransform);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, startNaiveHoughTransform, stopNaiveHoughTransform);

        std::cout << "Execution Time on the GPU: " << milliseconds << std::endl;
    }
    else if (method == HOUGH_TRANSFORM_NAIVE_KERNEL2)
    {
    naiveKernel2:
        std::cout << "Executing Hough Transform on the Local Accumulator Kernel 2" << std::endl;
        cudaEvent_t startNaiveHoughTransform, stopNaiveHoughTransform;
        cudaEventCreate(&startNaiveHoughTransform);
        cudaEventCreate(&stopNaiveHoughTransform);

        cudaEventRecord(startNaiveHoughTransform);

        dim3 mygrid(ceil(imageColumns / (BLOCK_DIMENSIONS * 1.0)), ceil(imageRows / (BLOCK_DIMENSIONS * 1.0)));
        dim3 myblock(BLOCK_DIMENSIONS, BLOCK_DIMENSIONS);

        hough_transform_kernel_naive<<<mygrid, myblock>>>(gpuImageBuffer, deviceRTable, imageRows, imageColumns, minimumRadius, maximumRadius, distance);

        // std::cout << "Hough Transform Naive Kernel Execution Complete" << std::endl;
        cudaEventRecord(stopNaiveHoughTransform);
        cudaEventSynchronize(stopNaiveHoughTransform);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, startNaiveHoughTransform, stopNaiveHoughTransform);

        std::cout << "Execution Time on the GPU: " << milliseconds << std::endl;
    }
    else if (method == HOUGH_TRANSFORM_SHARED_LOCAL_ACCUMULATOR)
    {
        std::cout << "Executing Hough Transform on the Shared Memory Kernel." << std::endl;
        if (sizeof(uchar) * imageRows * imageColumns > 64000 / 4.0)
        {
            printf("The image is too large to run using shared memory. Running on global memory kernel. Rows: %d, Columns: %d\n", imageRows, imageColumns);
            exit(-1);
        }
        cudaEvent_t startNaiveHoughTransform, stopNaiveHoughTransform;
        cudaEventCreate(&startNaiveHoughTransform);
        cudaEventCreate(&stopNaiveHoughTransform);

        cudaEventRecord(startNaiveHoughTransform);

        dim3 mygrid(ceil(imageColumns / (BLOCK_DIMENSIONS * 1.0)), ceil(imageRows / (BLOCK_DIMENSIONS * 1.0)));
        dim3 myblock(BLOCK_DIMENSIONS, BLOCK_DIMENSIONS);

        hough_transform_kernel_naive<<<mygrid, myblock, sizeof(uchar) * imageRows * imageColumns>>>(gpuImageBuffer, deviceRTable, imageRows, imageColumns, minimumRadius, maximumRadius, distance);

        // std::cout << "Hough Transform Naive Kernel Execution Complete" << std::endl;
        cudaEventRecord(stopNaiveHoughTransform);
        cudaEventSynchronize(stopNaiveHoughTransform);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, startNaiveHoughTransform, stopNaiveHoughTransform);

        std::cout << "Execution Time on the GPU: " << milliseconds << std::endl;
    }
    else
    {
        std::cout << "Invalid Kernel Method Chosen. | (method): " << method << std::endl;
    }

    /* Modications are not being made to the image, so no copy back to the host is required. */
    // Copy data from device to host
    unsigned int *hostRTable;
    hostRTable = (unsigned int *)malloc(imageColumns * imageRows * (maximumRadius - minimumRadius) * sizeof(unsigned int));
    err = cudaMemcpy(hostRTable, deviceRTable, imageColumns * imageRows * (maximumRadius - minimumRadius) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cudaFree(gpuImageBuffer);
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    parseRTable(circles, hostRTable, minimumRadius, maximumRadius, imageRows, imageColumns);

    // Free allocated memory
    cudaFree(deviceRTable);
    cudaFree(gpuImageBuffer);
    free(hostRTable);
    std::cout << "Cuda Memory Freed" << std::endl;

    std::cout << "GPU Hough Transform Execution Complete" << std::endl;
}

