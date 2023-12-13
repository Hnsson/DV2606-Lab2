/***************************************************************************
*
* Sequential version of Gauss-Jordan row reduction
*
***************************************************************************/

#include <stdio.h>
#include <sys/time.h>

#define MAX_SIZE 4096

typedef double matrix[MAX_SIZE][MAX_SIZE];

int	N;		        /* matrix size		*/
int	maxnum;		    /* max number of element*/
int	PRINT;		    /* print switch		*/
char* Init;		/* matrix init type	*/
matrix	A;		        /* matrix A		*/
double	b[MAX_SIZE];	/* vector b             */
double	y[MAX_SIZE];	/* vector y             */
double temp_d_a[MAX_SIZE*MAX_SIZE]; /* flattened matrix A */

__global__ void gauss_jordan_p1(double *d_A, double *d_b, double *d_y, int N, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i != 0) return; // Perform sequentially to not have race-condition
    
    for (int j = k + 1; j < N; j++) { // Division step
        d_A[k*N + j] = d_A[k*N + j] / d_A[k*N + k];
    }
    d_y[k] = d_b[k] / d_A[k*N + k];
    d_A[k*N + k] = 1.0;
}

__global__ void gauss_jordan_p2(double *d_A, double *d_b, double *d_y, int N, int k) {
    int i = threadIdx.x;
    int j = blockIdx.x;
    int stride = (N > 1024) ? 1024 : 0; // Stride due to maximum allowed threads per block on a device

    if (j != k) {
        if (i >= k + 1 && stride == 0) { // If NO stride is needed, perform oridinary calculations
            d_A[j * N + i] = d_A[j * N + i] - d_A[j * N + k] * d_A[k * N + i];
        }
        if (stride > 0) { // If stride is needed, perform same calculations over the stride
            for (int index = i; index < N; index += stride) {
                if (index >= k + 1) {
                    d_A[j * N + index] = d_A[j * N + index] - d_A[j * N + k] * d_A[k * N + index];
                }
            }
        }

        __syncthreads();

        if(j > k && i == 0) { // Perform calc based on blockId respected to k
            d_b[j] = d_b[j] - d_A[j * N + k] * d_y[k];
            d_A[j * N + k] = 0.0;
        } else if (j < k && i == 0) {
            d_y[j] = d_y[j] - d_A[j * N + k] * d_y[k];
            d_A[j * N + k] = 0.0;
        }

    }
}


/* forward declarations */
void print_matrix(void);
void init_default(void);
void print_options(void);
void read_options(int argc, char**argv);
void init_matrix(void);

int
main(int argc, char** argv)
{
    printf("Gauss Jordan\n");

    /* INIT SETTINGS */
    init_default();
    read_options(argc, argv);
    init_matrix();
    print_options();
    if (PRINT == 1)
        print_matrix();

    // Perform calculations using CUDA
    double *d_A;
    double *d_b;
    double *d_y;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_A, sizeof(double) * MAX_SIZE * MAX_SIZE);
    cudaMalloc((void**)&d_b, sizeof(double) * MAX_SIZE);
    cudaMalloc((void**)&d_y, sizeof(double) * MAX_SIZE);

    // Flatten the 2D matrix to a 1D array for CUDA usage
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            temp_d_a[i*N + j] = A[i][j];
        }
    }
 
    // Copy data from host to device
    cudaMemcpy(d_A, temp_d_a, sizeof(double) * MAX_SIZE * MAX_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(double) * MAX_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(double) * MAX_SIZE, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int threads = (N > 1024) ? 1024 : N; // If N > 1024 (allowed threads / block) we will stride with max (1024)
    int block = N;
 
    clock_t t;
    t = clock();
    // Call the CUDA kernel function
    for (int k = 0; k < N; k++) {
        gauss_jordan_p1<<<1, 1>>>(d_A, d_b, d_y, N, k); // Phase 1 - Divison
        gauss_jordan_p2<<<block, threads>>>(d_A, d_b, d_y, N, k); // Phase 2 - Both Elimination Steps
    }
 
    cudaMemcpy(b, d_b, sizeof(double) * MAX_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, sizeof(double) * MAX_SIZE, cudaMemcpyDeviceToHost);

    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
    printf("gauss jordan took %.3f seconds to execute \n", time_taken);
    
    // Copy the result back to the host
    cudaMemcpy(temp_d_a, d_A, sizeof(double) * MAX_SIZE * MAX_SIZE, cudaMemcpyDeviceToHost);

    // Unflatten array by copying over results from device array to matrix A
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = temp_d_a[i*N + j];
        }
    }

    // Free memory on the GPU
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_y);

    // Print the modified matrix and vectors
    if (PRINT == 1)
        print_matrix();

    return 0;
}

void
print_matrix(void)
{
    int i, j;

    printf("Matrix A:\n");
    for (i = 0; i < N; i++) {
        printf("[");
        for (j = 0; j < N; j++)
            printf(" %5.2f,", A[i][j]);
        printf("]\n");
    }
    printf("Vector y:\n[");
    for (j = 0; j < N; j++)
        printf(" %5.2f,", y[j]);
    printf("]\n");
    printf("\n\n");
}

void
init_default(void)
{
    N = 2048;
    maxnum = 15.0;
    PRINT = 0;
    char init[] = "rand";
    Init = init;
}

void
print_options(void) {
    printf("\nsize      = %dx%d ", N, N);
    printf("\nmaxnum    = %d \n", maxnum);
    printf("Init	  = %s \n", Init);
    printf("printing  = %s \n\n", (PRINT == 1) ? "true" : "false");
 
}

void
read_options(int argc, char** argv)
{
    char* prog;

    prog = *argv;
    while (++argv, --argc > 0)
        if (**argv == '-')
            switch (*++ * argv) {
            case 'n':
                --argc;
                N = atoi(*++argv);
                break;
            case 'h':
                printf("\nHELP: try sor -u \n\n");
                exit(0);
                break;
            case 'u':
                printf("\nUsage: gaussian [-n problemsize]\n");
                printf("           [-D] show default values \n");
                printf("           [-h] help \n");
                printf("           [-I init_type] fast/rand \n");
                printf("           [-m maxnum] max random no \n");
                printf("           [-P print_switch] 0/1 \n");
                exit(0);
                break;
            case 'D':
                printf("\nDefault:  n         = %d ", N);
                printf("\n          Init      = rand");
                printf("\n          maxnum    = 5 ");
                printf("\n          P         = 0 \n\n");
                exit(0);
                break;
            case 'I':
                --argc;
                Init = *++argv;
                break;
            case 'm':
                --argc;
                maxnum = atoi(*++argv);
                break;
            case 'P':
                --argc;
                PRINT = atoi(*++argv);
                break;
            default:
                printf("%s: ignored option: -%s\n", prog, *argv);
                printf("HELP: try %s -u \n\n", prog);
                break;
            }
}

void
init_matrix(void)
{
    int i, j;

    if (strcmp(Init, "rand") == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                if (i == j) /* diagonal dominance */
                    A[i][j] = (double)(rand() % maxnum) + 5.0;
                else
                    A[i][j] = (double)(rand() % maxnum) + 1.0;
            }
        }
    }
    if (strcmp(Init, "fast") == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                if (i == j) /* diagonal dominance */
                    A[i][j] = 5.0;
                else
                    A[i][j] = 2.0;
            }
        }
    }

    /* Initialize vectors b and y */
    for (i = 0; i < N; i++) {
        b[i] = 2.0;
        y[i] = 1.0;
    }
}
