#include<stdlib.h>
#include<time.h>
#include<stdio.h>

void serialConvolution(char input[], char mask[], char output[], int w, int h, int maskWidth);

__global__ void naiveConvolution(char input[], char mask[], char output[], int w, int h, int maskWidth);

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
    printf("Elapsed time for serial: %f", milliseconds);

}