#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define swap(A,B) {int temp=A;A=B;B=temp;}

// First kernel to perform odd phase of odd-even sort
__global__ void oddPhaseSort(int *arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i % 2 == 0 && i < n - 1) {
        if (arr[i] > arr[i + 1]) {
            swap(arr[i], arr[i + 1]);
        }
    }
}

// Second kernel to perform even phase of odd-even sort
__global__ void evenPhaseSort(int *arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i % 2 != 0 && i < n - 1) {
        if (arr[i] > arr[i + 1]) {
            swap(arr[i], arr[i + 1]);
        }
    }
}

// __global__ void oddEvenSort(int *arr, int n) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i % 2 == 0 && i < n - 1) {
//         if (arr[i] > arr[i + 1]) {
//             swap(arr[i], arr[i + 1]);
//         }
//     }

//     if (i % 2 != 0 && i < n - 1) {
//         if (arr[i] > arr[i + 1]) {
//             swap(arr[i], arr[i + 1]);
//         }
//     }
// }



void init_arr(int *arr, int size) {
    for (int i = 0; i < size; i++)
        arr[i] = rand() % (100 - 1);
}

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


int main() {
    int arraySize = 1 << 19; // Define array size
    int *array, *d_array;
    size_t size = arraySize * sizeof(int);

    // Allocate memory for host and device arrays
    array = (int*)malloc(size);

    // Allocate memory on the device
    cudaMalloc((void **)&d_array, arraySize * sizeof(int));

    init_arr(array, arraySize);
    // print_arr(array, arraySize);
    sorted(array, arraySize);

    // Copy input array from host to device
    cudaMemcpy(d_array, array, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with appropriate block and grid dimensions
    int T = 256;
    int B = ceil((float) arraySize / T);

    clock_t t; 
    t = clock(); 

    for (int phase = 0; phase < arraySize; ++phase) {
        if (phase % 2 == 0) {
            evenPhaseSort<<<B, T>>>(d_array, arraySize);
        } else {
            oddPhaseSort<<<B, T>>>(d_array, arraySize);
        }
    }

    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
    printf("sorting took %f seconds to execute \n", time_taken); 

    // Copy the result back from the device to the host
    cudaMemcpy(array, d_array, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Free memory on the device
    cudaFree(d_array);

    // print_arr(array, arraySize);
    sorted(array, arraySize);

    return 0;
}
