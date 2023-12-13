#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define swap(A,B) {int temp=A;A=B;B=temp;}

// Performing odd phase of odd-even sort
__global__ void oddEvenSort(int *arr, int n) {
    int i;

    for (i = 0; i < n; i++) {
        if (i % 2 == 0) {
            for (int j = threadIdx.x * 2; j < n - 1; j+= blockDim.x * 2)
                if (arr[j] > arr[j + 1])
                    swap(arr[j], arr[j + 1]);
        } else {
            for (int j = threadIdx.x * 2 + 1; j < n - 1; j+= blockDim.x * 2)
                if (arr[j] > arr[j + 1])
                    swap(arr[j], arr[j + 1]);
        }
        __syncthreads();
    }
}

// Init array with random values
void
init_arr(int *arr, int size)
{
    int i;
    for (i = 0; i < size; i++)
        arr[i] = rand() % (100- 1);
}

// Check if array is sorted
void
sorted(int *v, int size)
{
    int a = 1, d = 1, i = 0;

    while((a == 1 || d == 1) && i < size - 1) {
        if (v[i] < v[i+1])
            d = 0;
        else if (v[i] > v[i+1])
            a = 0;
        i++;
    }

    if (a == 1)
        printf("The array is sorted in ascending order.\n");
    else if (d == 1)
        printf("The array is sorted in descending order.\n");
    else
        printf("The array is not sorted.\n");
}

void
print_arr(int *v, int size)
{
    int i;
    for (i = 0; i < size; i++) {
        printf("%d, ", v[i]);
    }
}

int
main()
{
    int n = 1 << 19; // Define array size
    int *arr, *d_arr;
    size_t size = n * sizeof(int);

    // Allocate memory for host and device arrays
    arr = (int*)malloc(size);
    cudaMalloc(&d_arr, size);

    // Initialize the array with random values
    for (int i = 0; i < n; ++i) {
        arr[i] = rand() % 1000;
    }

    // Copy data from host to device
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    // Launch the kernel for the parallel version
    dim3 blockSize(1024); // Set block size #nr threads
    dim3 gridSize(1);    // Set grid sizew #nr blocks

    sorted(arr, n);

    clock_t t; 
    t = clock(); 

    oddEvenSort<<<gridSize, blockSize>>>(d_arr, n);
    cudaDeviceSynchronize();
    
    // Copy sorted data from device to host
    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);

    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
    printf("sorting took %f seconds to execute \n", time_taken); 

    sorted(arr, n);
    // print_arr(arr, n);

    // Free memory
    free(arr);
    cudaFree(d_arr);

    return 0;
}
