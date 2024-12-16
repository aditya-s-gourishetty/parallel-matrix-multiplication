#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>

#define MAX_THREADS 100000
int num_threads;

double **allocate_matrix_memory(int n)
{
    double *data = (double *)malloc(n * n * sizeof(double));
    double **matrix = (double **)malloc(n * sizeof(double *));

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        matrix[i] = &data[i * n];
    }
    return matrix;
}

// Function to free the 2D matrix
void free_matrix_memory(double **matrix)
{
    free(matrix[0]);
    free(matrix);
}

// Matrix addition A = B + C
void matrix_add(int n, double **A, double **B, double **C)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

// Matrix subtraction C = A - B
void matrix_sub(int n, double **A, double **B, double **C)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

// Standard matrix multiplication C = A * B
void parallel_matrix_mult(int n, double **A, double **B, double **C)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            C[i][j] = 0;
            for (int k = 0; k < n; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Standard matrix multiplication C = A * B
void sequential_matrix_mult(int n, double **A, double **B, double **C)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            C[i][j] = 0;
            for (int k = 0; k < n; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Strassen's recursive matrix multiplication
void strassen(int n, double **A, double **B, double **C, int cutoff)
{
    if (n <= cutoff)
    {
        parallel_matrix_mult(n, A, B, C);
        return;
    }

    int new_size = n / 2;

    // Allocate space for submatrices
    double **A11 = allocate_matrix_memory(new_size);
    double **A12 = allocate_matrix_memory(new_size);
    double **A21 = allocate_matrix_memory(new_size);
    double **A22 = allocate_matrix_memory(new_size);
    double **B11 = allocate_matrix_memory(new_size);
    double **B12 = allocate_matrix_memory(new_size);
    double **B21 = allocate_matrix_memory(new_size);
    double **B22 = allocate_matrix_memory(new_size);

    double **M1 = allocate_matrix_memory(new_size);
    double **M2 = allocate_matrix_memory(new_size);
    double **M3 = allocate_matrix_memory(new_size);
    double **M4 = allocate_matrix_memory(new_size);
    double **M5 = allocate_matrix_memory(new_size);
    double **M6 = allocate_matrix_memory(new_size);
    double **M7 = allocate_matrix_memory(new_size);

    double **M1temp1 = allocate_matrix_memory(new_size);
    double **M1temp2 = allocate_matrix_memory(new_size);

    double **M2temp1 = allocate_matrix_memory(new_size);

    double **M3temp1 = allocate_matrix_memory(new_size);

    double **M4temp1 = allocate_matrix_memory(new_size);

    double **M5temp1 = allocate_matrix_memory(new_size);

    double **M6temp1 = allocate_matrix_memory(new_size);
    double **M6temp2 = allocate_matrix_memory(new_size);

    double **M7temp1 = allocate_matrix_memory(new_size);
    double **M7temp2 = allocate_matrix_memory(new_size);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < new_size; i++)
    {
        for (int j = 0; j < new_size; j++)
        {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + new_size];
            A21[i][j] = A[i + new_size][j];
            A22[i][j] = A[i + new_size][j + new_size];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + new_size];
            B21[i][j] = B[i + new_size][j];
            B22[i][j] = B[i + new_size][j + new_size];
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            // Compute M1 to M7
            // M1 = (A11 + A22) * (B11 + B22)
            matrix_add(new_size, A11, A22, M1temp1);
            matrix_add(new_size, B11, B22, M1temp2);
            strassen(new_size, M1temp1, M1temp2, M1, cutoff);
        }

#pragma omp section
        {
            // M2 = (A21 + A22) * B11
            matrix_add(new_size, A21, A22, M2temp1);
            strassen(new_size, M2temp1, B11, M2, cutoff);
        }

#pragma omp section
        {
            // M3 = A11 * (B12 - B22)
            matrix_sub(new_size, B12, B22, M3temp1);
            strassen(new_size, A11, M3temp1, M3, cutoff);
        }

#pragma omp section
        {
            // M4 = A22 * (B21 - B11)
            matrix_sub(new_size, B21, B11, M4temp1);
            strassen(new_size, A22, M4temp1, M4, cutoff);
        }

#pragma omp section
        {
            // M5 = (A11 + A12) * B22
            matrix_add(new_size, A11, A12, M5temp1);
            strassen(new_size, M5temp1, B22, M5, cutoff);
        }

#pragma omp section
        {
            // M6 = (A21 - A11) * (B11 + B12)
            matrix_sub(new_size, A21, A11, M6temp1);
            matrix_add(new_size, B11, B12, M6temp2);
            strassen(new_size, M6temp1, M6temp2, M6, cutoff);
        }

#pragma omp section
        {
            // M7 = (A12 - A22) * (B21 + B22)
            matrix_sub(new_size, A12, A22, M7temp1);
            matrix_add(new_size, B21, B22, M7temp2);
            strassen(new_size, M7temp1, M7temp2, M7, cutoff);
        }
    }

// Combine M1 to M7 to get C
#pragma omp parallel for collapse(2)
    for (int i = 0; i < new_size; i++)
    {
        for (int j = 0; j < new_size; j++)
        {
            C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + new_size] = M3[i][j] + M5[i][j];
            C[i + new_size][j] = M2[i][j] + M4[i][j];
            C[i + new_size][j + new_size] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    }

    // Free memory
    free_matrix_memory(A11);
    free_matrix_memory(A12);
    free_matrix_memory(A21);
    free_matrix_memory(A22);
    free_matrix_memory(B11);
    free_matrix_memory(B12);
    free_matrix_memory(B21);
    free_matrix_memory(B22);
    free_matrix_memory(M1);
    free_matrix_memory(M2);
    free_matrix_memory(M3);
    free_matrix_memory(M4);
    free_matrix_memory(M5);
    free_matrix_memory(M6);
    free_matrix_memory(M7);
    free_matrix_memory(M1temp1);
    free_matrix_memory(M1temp2);
    free_matrix_memory(M2temp1);
    free_matrix_memory(M3temp1);
    free_matrix_memory(M4temp1);
    free_matrix_memory(M5temp1);
    free_matrix_memory(M6temp1);
    free_matrix_memory(M6temp2);
    free_matrix_memory(M7temp1);
    free_matrix_memory(M7temp2);
}

int areMatricesEqual(int n, double **C_serial, double **C)
{

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (C_serial[i][j] != C[i][j])
            {
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char *argv[])
{
    int n, n_prime, k, k_prime;
    struct timespec start, stop;

    if (argc != 4)
    {
        printf("Need three integers as input, num_threads can be < 0 which would result in default no. of threads \n");
        printf("Use:<executable_name> <k> <k'> <num_threads>\n");
        exit(0);
    }
    k = atoi(argv[argc - 3]);
    k_prime = atoi(argv[argc - 2]);
    num_threads = atoi(argv[argc - 1]);
    printf("---------------------------------------------\n");
    if (num_threads > 0)
    {
        printf("Number of threads: %d\n", num_threads);
        omp_set_num_threads(num_threads);
    }

    n = pow(2, k);
    n_prime = pow(2, k_prime);

    printf("---------------------------------------------\n");
    printf("Matrix size: %d x %d\n", n, n);
    printf("Cutoff size: %d x %d\n", n_prime, n_prime);
    // printf("Number of threads: %d\n", num_threads);

    double **A = allocate_matrix_memory(n);
    double **B = allocate_matrix_memory(n);
    double **C = allocate_matrix_memory(n);

// Initialize matrices A and B with random values
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }

    clock_gettime(CLOCK_REALTIME, &start);
    strassen(n, A, B, C, n_prime);
    clock_gettime(CLOCK_REALTIME, &stop);
    double total_time;
    total_time = (stop.tv_sec - start.tv_sec) + 0.000000001 * (stop.tv_nsec - start.tv_nsec);

    printf("Elapsed time for strassen's algorithm: %f seconds\n", total_time);

    double **C_serial = allocate_matrix_memory(n);
    clock_gettime(CLOCK_REALTIME, &start);
    sequential_matrix_mult(n, A, B, C_serial);
    clock_gettime(CLOCK_REALTIME, &stop);
    total_time = (stop.tv_sec - start.tv_sec) + 0.000000001 * (stop.tv_nsec - start.tv_nsec);

    printf("Elapsed time for serial matrix multiplication: %f seconds\n", total_time);

    if (areMatricesEqual(n, C_serial, C))
    {
        printf("Matrices are equal\n");
    }
    else
    {
        printf("Resultant Matrices are not equal for strassen's and naive serial matrix multiplication\n");
    }

    double **C_parallel = allocate_matrix_memory(n);
    clock_gettime(CLOCK_REALTIME, &start);
    parallel_matrix_mult(n, A, B, C_parallel);
    clock_gettime(CLOCK_REALTIME, &stop);
    total_time = (stop.tv_sec - start.tv_sec) + 0.000000001 * (stop.tv_nsec - start.tv_nsec);

    printf("Elapsed time for standard parallel matrix multiplication: %f seconds\n", total_time);

    // Free memory
    free_matrix_memory(A);
    free_matrix_memory(B);
    free_matrix_memory(C);
    printf("\n");

    return 0;
}

