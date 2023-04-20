#include<stdlib.h>
#include<time.h>
#include<stdio.h>
#include <cuda_runtime.h>
#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

void serialConvolution(char input[], char mask[], char output[], int rows, int cols, int maskWidth);

//NAIVE 2D CONVOLUTION
__global__ void naiveConvolution(char input[], char mask[], char output[], int rows, int cols, int maskWidth){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int pixVal = 0;

    if(row < rows && col < cols){
        int startCol = col - maskWidth / 2;
        int startRow = row - maskWidth / 2;

        for(int j = 0; j < maskWidth; j++){
            for(int k = 0; k < maskWidth; k++){
                int curRow = startRow + j;
                int curCol = startCol + k;

                if(curRow > -1 && curRow < rows && curCol > -1 && curCol < cols){
                    pixVal += input[curRow * cols + curCol] * mask[j * maskWidth + k];
                }
            }
        }
    }

    output[row * cols + col] = pixVal;
    
}

//MAIN FUNCTION
int main(void){
    srand(time(NULL)); //Set up random values
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //We will compute a convolution image on a 720p made up image
    char inputImage[720][1280];
    char outputImage[720][1280];
    char mask3x3[3][3];
    char mask5x5[5][5];

    //Initialize input image 
    for(int i = 0; i < 720; i++){
        for(int j = 0; j < 1280; j++){
            inputImage[i][j] = rand();
        }
    }

    //Initialize 3x3 mask and 5x5 mask
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            mask3x3[i][j] = rand();
        }
    }
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            mask5x5[i][j] = rand();
        }
    }

    //Apply convolution kernel to test values for accuracy
    //Will be using a 5x5 for now
    int kCols = 5;
    int kRows = 5;
    int kCenterX = kCols / 2;
    int kCenterY = kRows / 2;

    cudaEventRecord(start);
    for(int i=0; i < 720; ++i)              // rows
    {
        for(int j=0; j < 1280; ++j)          // columns
        {
            for(int m=0; m < kRows; ++m)     // kernel rows
            {
                int mm = kRows - 1 - m;      // row index of flipped kernel

                for(int n=0; n < kCols; ++n) // kernel columns
                {
                    int nn = kCols - 1 - n;  // column index of flipped kernel

                    // index of input signal, used for checking boundary
                    int ii = i + (kCenterY - mm);
                    int jj = j + (kCenterX - nn);

                    // ignore input samples which are out of bound
                    if( ii >= 0 && ii < 720 && jj >= 0 && jj < 1280 )
                        outputImage[i][j] += inputImage[ii][jj] * mask5x5[mm][nn];
                }
            }
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time for serial: %f\n", milliseconds);

    //Now we will do a naive implementation on CUDA
    char* hostInput;
    char* hostMask;
    char* hostOutput;
    char* deviceInput;
    char* deviceMask;
    char* deviceOutput;
    int rows = 720;
    int cols = 1280;
    int maskVal = 5;
    cudaEvent_t startNaive, stopNaive;
    cudaEventCreate(&startNaive);
    cudaEventCreate(&stopNaive);

    //Allocate Memory on host side
    hostInput = (char*)malloc(rows * cols * sizeof(char));
    hostMask = (char*)malloc(maskVal * maskVal * sizeof(char));
    hostOutput = (char*)malloc(rows * cols * sizeof(char));

    //Populare arrays on the host side
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            hostInput[i * rows + j] = inputImage[i][j];
        }
    }
    for(int i = 0; i < maskVal; i++){
        for(int j = 0; j < maskVal; j++){
            hostInput[i * rows + j] = mask5x5[i][j];
        }
    }

    //Allocate GPU memory here
    CUDA_CHECK(
        cudaMalloc((void **)&deviceInput, rows * cols * sizeof(char)));
    CUDA_CHECK(
        cudaMalloc((void **)&deviceMask, maskVal * maskVal * sizeof(char)));
    CUDA_CHECK(
        cudaMalloc((void **)&deviceOutput, rows * cols * sizeof(char)));
    CUDA_CHECK(cudaDeviceSynchronize());

    //Populate arrays on device side
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput,
                            rows * cols * sizeof(char),
                            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceMask, hostMask,
                            maskVal * maskVal * sizeof(char),
                            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    //Call the naive kernel
    cudaEventRecord(startNaive);
    int blockSize = 32;
    dim3 blockDim(blockSize,blockSize), gridDim( 1 + (cols - 1) / blockSize,
                                             1 + (rows - 1) / blockSize);
    naiveConvolution<<<gridDim, blockDim>>>
        (deviceInput, deviceMask, deviceOutput, rows, cols, maskVal);
    cudaEventRecord(stopNaive);
    cudaEventSynchronize(stopNaive);
    //CUDA_CHECK(cudaGetLastError());
    //CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hostOutput, deviceOutput,
                            rows * cols * sizeof(char),
                            cudaMemcpyDeviceToHost));
    
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startNaive, stopNaive);
    printf("Elapsed time for naive: %f\n", milliseconds);


    //Memory Freeing
    //CUDA
    cudaFree(deviceInput);
    cudaFree(deviceMask);
    cudaFree(deviceOutput);

    //CPU
    free(hostInput);
    free(hostMask);
    free(hostOutput);
}